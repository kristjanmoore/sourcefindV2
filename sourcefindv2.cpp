#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void error_input_message(const std::string &file_name) {
  std::cerr << "Something wrong with input file " << file_name
            << ". See below. Exiting..." << std::endl;
  std::exit(EXIT_FAILURE);
}

std::string trim(const std::string &s) {
  std::string::size_type first = s.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return "";
  std::string::size_type last = s.find_last_not_of(" \t\r\n");
  return s.substr(first, last - first + 1);
}

std::vector<std::string> split_whitespace(const std::string &line) {
  std::istringstream iss(line);
  std::vector<std::string> tokens;
  std::string token;
  while (iss >> token) {
    tokens.push_back(token);
  }
  return tokens;
}

std::vector<std::string> unique_preserve(const std::vector<std::string> &values) {
  std::vector<std::string> result;
  std::set<std::string> seen;
  for (const auto &v : values) {
    if (!seen.count(v)) {
      seen.insert(v);
      result.push_back(v);
    }
  }
  return result;
}

std::vector<std::string> line_check(const std::vector<std::string> &lines,
                                    std::size_t skip, const std::string &match,
                                    const std::string &file_name) {
  if (skip >= lines.size()) {
    error_input_message(file_name);
  }
  auto tokens = split_whitespace(lines[skip]);
  if (tokens.empty() || tokens[0] != match) {
    error_input_message(file_name);
  }
  tokens.erase(tokens.begin());
  return tokens;
}

struct Parameters {
  int self_copy_ind = 0;
  int num_pops = 0;
  int surr_exp = 0;
  std::string id_file;
  std::string copyvector_file;
  std::string save_file;
  std::vector<std::string> donor_pops_all;
  std::vector<std::string> surrogate_popnames;
  std::vector<std::string> target_popnames;
  int num_slots = 0;
  int num_runs = 0;
  int burn_in = 0;
  int thin_val = 0;
};

struct IdEntry {
  std::string individual;
  std::string population;
  std::string include_flag;
};

std::vector<IdEntry> read_id_file(const std::string &path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Unable to open id file: " + path);
  }
  std::vector<IdEntry> entries;
  std::string line;
  while (std::getline(in, line)) {
    line = trim(line);
    if (line.empty()) continue;
    auto tokens = split_whitespace(line);
    if (tokens.size() < 3) {
      throw std::runtime_error("Invalid line in id file: " + line);
    }
    entries.push_back({tokens[0], tokens[1], tokens[2]});
  }
  return entries;
}

struct CopyvectorData {
  std::vector<std::string> row_names;
  std::vector<std::string> col_names;
  std::vector<std::vector<double>> matrix;
};

CopyvectorData read_copyvector_file(const std::string &path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Unable to open copyvector file: " + path);
  }
  std::string line;
  if (!std::getline(in, line)) {
    throw std::runtime_error("Copyvector file is empty: " + path);
  }
  auto header_tokens = split_whitespace(line);
  if (header_tokens.empty()) {
    throw std::runtime_error("Copyvector header missing columns: " + path);
  }
  CopyvectorData data;
  data.col_names.assign(header_tokens.begin() + 1, header_tokens.end());

  while (std::getline(in, line)) {
    line = trim(line);
    if (line.empty()) continue;
    auto tokens = split_whitespace(line);
    if (tokens.size() < 1 + data.col_names.size()) {
      throw std::runtime_error("Row in copyvector file has insufficient columns: " + line);
    }
    data.row_names.push_back(tokens[0]);
    std::vector<double> row(data.col_names.size());
    for (std::size_t i = 0; i < data.col_names.size(); ++i) {
      row[i] = std::stod(tokens[i + 1]);
    }
    data.matrix.push_back(std::move(row));
  }
  return data;
}

template <typename T>
std::vector<T> set_difference_vec(const std::vector<T> &a,
                                  const std::vector<T> &b) {
  std::set<T> set_b(b.begin(), b.end());
  std::vector<T> result;
  for (const auto &val : a) {
    if (!set_b.count(val)) {
      result.push_back(val);
    }
  }
  return result;
}

template <typename T>
std::vector<T> unique_vector(const std::vector<T> &vec) {
  std::vector<T> copy = vec;
  std::sort(copy.begin(), copy.end());
  copy.erase(std::unique(copy.begin(), copy.end()), copy.end());
  return copy;
}

double multi_prob_func(const std::vector<double> &x,
                       const std::vector<double> &target_vec) {
  double total = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    total += target_vec[i] * std::log(x[i]);
  }
  return total;
}

std::vector<double> col_means(const std::vector<std::vector<double>> &matrix,
                              const std::vector<int> &row_indices,
                              std::size_t num_cols) {
  std::vector<double> sums(num_cols, 0.0);
  for (int idx : row_indices) {
    const auto &row = matrix[idx];
    for (std::size_t j = 0; j < num_cols; ++j) {
      sums[j] += row[j];
    }
  }
  if (!row_indices.empty()) {
    double denom = static_cast<double>(row_indices.size());
    for (double &val : sums) {
      val /= denom;
    }
  }
  return sums;
}

std::vector<int> sample_indices(std::mt19937_64 &rng,
                                const std::vector<int> &pool,
                                int sample_size, bool replace) {
  if (pool.empty()) return {};
  if (replace) {
    std::vector<int> result(sample_size);
    std::uniform_int_distribution<std::size_t> dist(0, pool.size() - 1);
    for (int i = 0; i < sample_size; ++i) {
      result[i] = pool[dist(rng)];
    }
    return result;
  }
  if (sample_size >= static_cast<int>(pool.size())) {
    return pool;
  }
  std::vector<int> copy = pool;
  for (int i = 0; i < sample_size; ++i) {
    std::uniform_int_distribution<std::size_t> dist(i, copy.size() - 1);
    std::size_t j = dist(rng);
    std::swap(copy[i], copy[j]);
  }
  copy.resize(sample_size);
  return copy;
}

int sample_one_with_prob(std::mt19937_64 &rng, const std::vector<int> &values,
                         const std::vector<double> &probabilities) {
  std::discrete_distribution<int> dist(probabilities.begin(), probabilities.end());
  int idx = dist(rng);
  return values[idx];
}

std::vector<double> tabulate(const std::vector<int> &values, int nbins) {
  std::vector<double> counts(nbins, 0.0);
  for (int v : values) {
    if (v >= 0 && v < nbins) {
      counts[v] += 1.0;
    }
  }
  return counts;
}

void write_results(const std::string &save_file, const std::string &target_name,
                   const std::vector<double> &multi_prob_final,
                   const std::vector<std::vector<double>> &surr_tochoose_final,
                   const std::vector<std::string> &surr_pops,
                   const std::vector<std::string> &surr_mat_row_names,
                   int num_slots) {
  (void)num_slots;
  std::ofstream out(save_file, std::ios::app);
  if (!out) {
    throw std::runtime_error("Unable to open output file: " + save_file);
  }

  std::unordered_map<std::string, std::size_t> pop_index;
  for (std::size_t i = 0; i < surr_pops.size(); ++i) {
    pop_index.emplace(surr_pops[i], i);
  }

  std::vector<std::vector<double>> to_print(multi_prob_final.size(),
                                            std::vector<double>(surr_pops.size(), 0.0));
  for (std::size_t row = 0; row < surr_tochoose_final.size(); ++row) {
    for (std::size_t j = 0; j < surr_mat_row_names.size(); ++j) {
      auto it = pop_index.find(surr_mat_row_names[j]);
      if (it != pop_index.end()) {
        to_print[row][it->second] = surr_tochoose_final[row][j];
      }
    }
  }

  out << "target\tposterior.prob";
  for (const auto &pop : surr_pops) {
    out << '\t' << pop;
  }
  out << '\n';

  out << std::setprecision(15);
  for (std::size_t i = 0; i < multi_prob_final.size(); ++i) {
    out << target_name << '\t' << multi_prob_final[i];
    for (double val : to_print[i]) {
      out << '\t' << val;
    }
    out << '\n';
  }
}

void sourcefind(const std::vector<double> &target_vec,
                const std::vector<std::vector<double>> &surr_mat,
                int num_pops, int surr_exp, int num_slots, int num_runs,
                int burn_in, int thin_val, const std::string &target_name,
                const std::vector<std::string> &surr_pops,
                const std::vector<std::string> &surr_mat_row_names,
                const std::string &save_file, std::mt19937_64 &rng) {
  bool add_prior = true;
  int num_surr = static_cast<int>(surr_mat.size());
  if (num_surr == 0) {
    throw std::runtime_error("No surrogate rows available after filtering.");
  }
  if (num_pops > num_surr) num_pops = num_surr;
  if (surr_exp > num_surr) surr_exp = num_surr;
  if (num_slots < num_pops) num_pops = num_slots;
  if (num_slots < surr_exp) surr_exp = num_slots;
  int pops_to_replace = 1;

  std::vector<double> prior_prob(num_pops, 0.0);
  double factorial = 1.0;
  double denom = 0.0;
  for (int i = 0; i < num_pops; ++i) {
    factorial *= static_cast<double>(i + 1);
    prior_prob[i] = std::pow(static_cast<double>(surr_exp), static_cast<double>(i + 1)) / factorial;
    denom += prior_prob[i];
  }
  for (double &val : prior_prob) {
    val /= denom;
  }

  std::vector<int> initial_pool(num_surr);
  std::iota(initial_pool.begin(), initial_pool.end(), 0);
  auto surr_tochoose_orig = sample_indices(rng, initial_pool, num_pops, true);
  auto surr_tochoose = sample_indices(rng, surr_tochoose_orig, num_slots, true);

  auto current_means = col_means(surr_mat, surr_tochoose, target_vec.size());
  double prev_multinom = multi_prob_func(current_means, target_vec);
  if (add_prior && !prior_prob.empty()) {
    auto unique_vals = unique_vector(surr_tochoose);
    prev_multinom += std::log(prior_prob[unique_vals.size() - 1]);
  }
  std::vector<int> surr_tochoose_prev = surr_tochoose;
  std::vector<std::vector<double>> surr_tochoose_final;
  std::vector<double> multi_prob_final;
  int num_sample = 0;
  std::uniform_real_distribution<double> uniform01(0.0, 1.0);

  for (int m = 1; m <= num_runs; ++m) {
    auto unique_vals = unique_vector(surr_tochoose);
    std::vector<double> replace_probs(unique_vals.size(),
                                      1.0 / static_cast<double>(num_slots));
    std::vector<int> pops_to_replace_vec(pops_to_replace);
    for (int i = 0; i < pops_to_replace; ++i) {
      pops_to_replace_vec[i] = sample_one_with_prob(rng, unique_vals, replace_probs);
    }

    for (int idx = 0; idx < pops_to_replace; ++idx) {
      int pop = pops_to_replace_vec[idx];
      std::vector<int> pop_slots;
      for (std::size_t pos = 0; pos < surr_tochoose.size(); ++pos) {
        if (surr_tochoose[pos] == pop) {
          pop_slots.push_back(static_cast<int>(pos));
        }
      }
      if (pop_slots.empty()) {
        continue;
      }
      std::uniform_int_distribution<int> slot_count_dist(1, static_cast<int>(pop_slots.size()));
      int num_slots_to_replace_final = slot_count_dist(rng);
      double random_val1 = uniform01(rng);
      if (random_val1 < 0.1) {
        num_slots_to_replace_final = static_cast<int>(pop_slots.size());
      }
      auto slots_to_replace = sample_indices(rng, pop_slots, num_slots_to_replace_final, false);
      std::vector<int> all_slots(surr_tochoose.size());
      std::iota(all_slots.begin(), all_slots.end(), 0);
      std::vector<int> slots_to_keep;
      slots_to_keep.reserve(all_slots.size() - slots_to_replace.size());
      std::set<int> replace_set(slots_to_replace.begin(), slots_to_replace.end());
      for (int slot : all_slots) {
        if (!replace_set.count(slot)) {
          slots_to_keep.push_back(slot);
        }
      }

      std::vector<int> all_surr;
      if (!slots_to_keep.empty()) {
        std::vector<int> keep_values;
        keep_values.reserve(slots_to_keep.size());
        for (int slot : slots_to_keep) {
          keep_values.push_back(surr_tochoose[slot]);
        }
        all_surr = unique_vector(keep_values);
      }
      if (static_cast<int>(all_surr.size()) < num_pops) {
        std::vector<int> all_options(num_surr);
        std::iota(all_options.begin(), all_options.end(), 0);
        auto remaining = set_difference_vec(all_options, all_surr);
        int needed = num_pops - static_cast<int>(all_surr.size());
        if (needed > 0 && !remaining.empty()) {
          auto sampled = sample_indices(rng, remaining,
                                        std::min(needed, static_cast<int>(remaining.size())),
                                        false);
          all_surr.insert(all_surr.end(), sampled.begin(), sampled.end());
        }
      }
      std::vector<int> other_surr = set_difference_vec(all_surr, std::vector<int>{pop});
      if (random_val1 < 0.1) {
        std::vector<int> all_options(num_surr);
        std::iota(all_options.begin(), all_options.end(), 0);
        std::vector<int> exclude;
        exclude.reserve(slots_to_keep.size() + 1);
        for (int slot : slots_to_keep) {
          exclude.push_back(surr_tochoose[slot]);
        }
        exclude.push_back(pop);
        auto tmp = set_difference_vec(all_options, unique_vector(exclude));
        other_surr = tmp;
      }
      if (other_surr.empty()) {
        std::vector<int> all_options(num_surr);
        std::iota(all_options.begin(), all_options.end(), 0);
        auto tmp = set_difference_vec(all_options, std::vector<int>{pop});
        if (tmp.empty()) {
          continue;
        }
        other_surr = tmp;
        slots_to_replace.clear();
        slots_to_replace.reserve(num_slots);
        for (int slot = 0; slot < num_slots; ++slot) {
          slots_to_replace.push_back(slot);
        }
      }
      std::uniform_int_distribution<std::size_t> other_dist(0, other_surr.size() - 1);
      int replacement = other_surr[other_dist(rng)];
      for (int slot : slots_to_replace) {
        surr_tochoose[slot] = replacement;
      }
    }

    current_means = col_means(surr_mat, surr_tochoose, target_vec.size());
    double new_multinom = multi_prob_func(current_means, target_vec);
    if (add_prior && !prior_prob.empty()) {
      auto unique_vals_new = unique_vector(surr_tochoose);
      new_multinom += std::log(prior_prob[unique_vals_new.size() - 1]);
    }
    double accept_prob = std::min(1.0, std::exp(new_multinom - prev_multinom));
    double random_val = uniform01(rng);
    if (random_val <= accept_prob) {
      prev_multinom = new_multinom;
      surr_tochoose_prev = surr_tochoose;
    } else {
      surr_tochoose = surr_tochoose_prev;
    }
    if (m == (burn_in + 1 + num_sample * thin_val)) {
      multi_prob_final.push_back(prev_multinom);
      auto counts = tabulate(surr_tochoose, num_surr);
      for (double &val : counts) {
        val /= static_cast<double>(num_slots);
      }
      surr_tochoose_final.push_back(std::move(counts));
      ++num_sample;
    }
  }

  write_results(save_file, target_name, multi_prob_final, surr_tochoose_final,
                surr_pops, surr_mat_row_names, num_slots);
}

}  // namespace

int main(int argc, char **argv) {
  const struct option long_options[] = {
      {"chunklengths", required_argument, nullptr, 'c'},
      {"parameters", required_argument, nullptr, 'p'},
      {"target", required_argument, nullptr, 't'},
      {"output", required_argument, nullptr, 'o'},
      {"idfile", required_argument, nullptr, 'i'},
      {nullptr, 0, nullptr, 0}};

  std::string chunklengths_opt;
  std::string parameters_opt;
  std::string target_opt;
  std::string output_opt;
  std::string idfile_opt;

  while (true) {
    int option_index = 0;
    int c = getopt_long(argc, argv, "c:p:t:o:i:", long_options, &option_index);
    if (c == -1) break;
    switch (c) {
      case 'c':
        chunklengths_opt = optarg;
        break;
      case 'p':
        parameters_opt = optarg;
        break;
      case 't':
        target_opt = optarg;
        break;
      case 'o':
        output_opt = optarg;
        break;
      case 'i':
        idfile_opt = optarg;
        break;
      default:
        std::cerr << "Error parsing command line options" << std::endl;
        return EXIT_FAILURE;
    }
  }

  if (parameters_opt.empty()) {
    std::cerr << "Error. Must provide valid parameter infile" << std::endl;
    return EXIT_FAILURE;
  }

  std::ifstream param_in(parameters_opt);
  if (!param_in) {
    std::cerr << "Unable to open parameter infile: " << parameters_opt << std::endl;
    return EXIT_FAILURE;
  }
  std::vector<std::string> param_lines;
  std::string line;
  while (std::getline(param_in, line)) {
    param_lines.push_back(trim(line));
  }

  Parameters params;
  try {
    auto self_copy_tokens = line_check(param_lines, 0, "self.copy.ind:", parameters_opt);
    if (self_copy_tokens.size() != 1) error_input_message(parameters_opt);
    params.self_copy_ind = std::stoi(self_copy_tokens[0]);
    if (!(params.self_copy_ind == 0 || params.self_copy_ind == 1)) {
      error_input_message(parameters_opt);
    }

    auto num_pops_tokens = line_check(param_lines, 1, "num.surrogates:", parameters_opt);
    if (num_pops_tokens.size() != 1) error_input_message(parameters_opt);
    params.num_pops = std::stoi(num_pops_tokens[0]);
    if (params.num_pops <= 1) error_input_message(parameters_opt);

    auto surr_exp_tokens = line_check(param_lines, 2, "exp.num.surrogates:", parameters_opt);
    if (surr_exp_tokens.size() != 1) error_input_message(parameters_opt);
    params.surr_exp = std::stoi(surr_exp_tokens[0]);
    if (params.surr_exp <= 0) error_input_message(parameters_opt);

    if (idfile_opt.empty()) {
      std::cout << "No idfile specified on the command line. Using value specified in paramfile" << std::endl;
      params.id_file = line_check(param_lines, 3, "input.file.ids:", parameters_opt)[0];
    } else {
      std::cout << "Using idfile file from the command line" << std::endl;
      params.id_file = idfile_opt;
    }

    if (chunklengths_opt.empty()) {
      std::cout << "No copyvector file specified on the command line. Using value specified in paramfile" << std::endl;
      params.copyvector_file = line_check(param_lines, 4, "input.file.copyvectors:", parameters_opt)[0];
    } else {
      std::cout << "Using copyvector file from the command line" << std::endl;
      params.copyvector_file = chunklengths_opt;
    }

    if (output_opt.empty()) {
      std::cout << "No output file name specified on the command line. Using populations specified in paramfile" << std::endl;
      params.save_file = line_check(param_lines, 5, "save.file:", parameters_opt)[0];
    } else {
      std::cout << "Using save.file file from the command line" << std::endl;
      params.save_file = output_opt;
    }

    auto donor_tokens = line_check(param_lines, 6, "copyvector.popnames:", parameters_opt);
    params.donor_pops_all = unique_preserve(donor_tokens);
    if (params.donor_pops_all.size() <= 1) error_input_message(parameters_opt);

    auto surrogate_tokens = line_check(param_lines, 7, "surrogate.popnames:", parameters_opt);
    params.surrogate_popnames = surrogate_tokens;
    auto surrogate_unique = unique_preserve(surrogate_tokens);
    if (surrogate_unique.size() <= 1) error_input_message(parameters_opt);

    if (target_opt.empty()) {
      std::cout << "No targets specified on the command line. Using populations specified in paramfile" << std::endl;
      params.target_popnames = line_check(param_lines, 8, "target.popnames:", parameters_opt);
    } else {
      std::cout << "Using targets file from the command line" << std::endl;
      params.target_popnames = {target_opt};
    }

    auto num_slots_tokens = line_check(param_lines, 9, "num.slots:", parameters_opt);
    if (num_slots_tokens.size() != 1) error_input_message(parameters_opt);
    params.num_slots = std::stoi(num_slots_tokens[0]);
    if (params.num_slots <= 1) error_input_message(parameters_opt);

    auto num_runs_tokens = line_check(param_lines, 10, "num.iterations:", parameters_opt);
    if (num_runs_tokens.size() != 1) error_input_message(parameters_opt);
    params.num_runs = std::stoi(num_runs_tokens[0]);
    if (params.num_runs <= 1) error_input_message(parameters_opt);

    auto burn_in_tokens = line_check(param_lines, 11, "num.burnin:", parameters_opt);
    if (burn_in_tokens.size() != 1) error_input_message(parameters_opt);
    params.burn_in = std::stoi(burn_in_tokens[0]);
    if (params.burn_in <= 1) error_input_message(parameters_opt);

    auto thin_tokens = line_check(param_lines, 12, "num.thin:", parameters_opt);
    if (thin_tokens.size() != 1) error_input_message(parameters_opt);
    params.thin_val = std::stoi(thin_tokens[0]);
    if (params.thin_val <= 1) error_input_message(parameters_opt);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  if (params.burn_in >= params.num_runs) {
    std::cerr << "You have specified more burn-in iterations than total iterations. Exiting...." << std::endl;
    return EXIT_FAILURE;
  }
  if ((params.burn_in + params.thin_val) > params.num_runs) {
    std::cerr << "Your specified burn-in iterations, total iterations, and thinning value will result in no samples. Exiting...." << std::endl;
    return EXIT_FAILURE;
  }
  if (params.surr_exp > params.num_pops) {
    std::cerr << "Your specified mean number of surrogates per iteration is greater than the total number of allowed surrogates per iteration. Exiting...." << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<IdEntry> id_entries;
  try {
    id_entries = read_id_file(params.id_file);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  std::vector<IdEntry> filtered_ids;
  for (const auto &entry : id_entries) {
    if (entry.include_flag != "0") {
      filtered_ids.push_back(entry);
    }
  }
  if (filtered_ids.empty()) {
    std::cerr << "SOMETHING WRONG WITH " << params.id_file << " -- NO NON-EXCLUDED INDS? Exiting...." << std::endl;
    return EXIT_FAILURE;
  }

  CopyvectorData copy_data;
  try {
    copy_data = read_copyvector_file(params.copyvector_file);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  std::unordered_map<std::string, std::vector<int>> col_lookup;
  for (std::size_t j = 0; j < copy_data.col_names.size(); ++j) {
    col_lookup[copy_data.col_names[j]].push_back(static_cast<int>(j));
  }
  std::unordered_map<std::string, std::vector<int>> row_lookup;
  for (std::size_t i = 0; i < copy_data.row_names.size(); ++i) {
    row_lookup[copy_data.row_names[i]].push_back(static_cast<int>(i));
  }

  auto donor_pops_all2 = unique_preserve(params.surrogate_popnames);

  auto ensure_in_lookup = [&](const std::vector<std::string> &names,
                              const std::string &context,
                              const std::unordered_map<std::string, std::vector<int>> &lookup_primary,
                              const std::unordered_map<std::string, std::vector<int>> &lookup_secondary,
                              const std::vector<IdEntry> &ids) {
    for (const auto &name : names) {
      bool in_ids = std::any_of(ids.begin(), ids.end(), [&](const IdEntry &e) {
        return e.population == name;
      });
      if (!in_ids && !lookup_primary.count(name)) {
        bool found_secondary = lookup_secondary.count(name) > 0;
        if (!found_secondary) {
          std::cerr << context << " " << name << " NOT FOUND IN " << params.id_file
                    << " OR " << (context == "COPY VECTOR COLUMN LABEL" ? "COLUMNS" : "ROWS")
                    << " OF " << params.copyvector_file << "! Exiting...." << std::endl;
          std::exit(EXIT_FAILURE);
        }
      }
    }
  };

  ensure_in_lookup(params.donor_pops_all, "COPY VECTOR COLUMN LABEL",
                   col_lookup, row_lookup, filtered_ids);
  ensure_in_lookup(donor_pops_all2, "SURROGATE POPULATION",
                   row_lookup, col_lookup, filtered_ids);
  ensure_in_lookup(params.target_popnames, "TARGET POPULATION",
                   row_lookup, col_lookup, filtered_ids);

  std::vector<std::vector<double>> predmat_orig(copy_data.matrix.size(),
                                                std::vector<double>(params.donor_pops_all.size(), 0.0));
  for (std::size_t pop_idx = 0; pop_idx < params.donor_pops_all.size(); ++pop_idx) {
    const auto &pop = params.donor_pops_all[pop_idx];
    std::vector<std::string> id_labels;
    id_labels.push_back(pop);
    for (const auto &entry : filtered_ids) {
      if (entry.population == pop) {
        id_labels.push_back(entry.individual);
      }
    }
    std::vector<int> matches;
    for (const auto &label : id_labels) {
      auto it = col_lookup.find(label);
      if (it != col_lookup.end()) {
        matches.insert(matches.end(), it->second.begin(), it->second.end());
      }
    }
    if (matches.empty()) {
      std::cerr << "NO INDS OF " << pop << " FOUND AMONG COLUMNS OF "
                << params.copyvector_file << "! Exiting...." << std::endl;
      return EXIT_FAILURE;
    }
    for (std::size_t row_idx = 0; row_idx < copy_data.matrix.size(); ++row_idx) {
      double sum = 0.0;
      for (int col_idx : matches) {
        sum += copy_data.matrix[row_idx][col_idx];
      }
      predmat_orig[row_idx][pop_idx] = sum;
    }
  }

  std::vector<std::vector<double>> predmat(donor_pops_all2.size(),
                                           std::vector<double>(params.donor_pops_all.size(), 0.0));
  for (std::size_t pop_idx = 0; pop_idx < donor_pops_all2.size(); ++pop_idx) {
    const auto &pop = donor_pops_all2[pop_idx];
    std::vector<std::string> id_labels;
    id_labels.push_back(pop);
    for (const auto &entry : filtered_ids) {
      if (entry.population == pop) {
        id_labels.push_back(entry.individual);
      }
    }
    std::vector<int> matches;
    for (const auto &label : id_labels) {
      auto it = row_lookup.find(label);
      if (it != row_lookup.end()) {
        matches.insert(matches.end(), it->second.begin(), it->second.end());
      }
    }
    if (matches.empty()) {
      std::cerr << "NO INDS OF " << pop << " FOUND AMONG ROWS OF "
                << params.copyvector_file << "! Exiting...." << std::endl;
      return EXIT_FAILURE;
    }
    for (std::size_t col_idx = 0; col_idx < params.donor_pops_all.size(); ++col_idx) {
      double sum = 0.0;
      for (int row_idx : matches) {
        sum += predmat_orig[row_idx][col_idx];
      }
      predmat[pop_idx][col_idx] = sum / static_cast<double>(matches.size());
    }
  }

  std::vector<std::vector<double>> recipient_mat(params.target_popnames.size(),
                                                 std::vector<double>(params.donor_pops_all.size(), 0.0));
  for (std::size_t target_idx = 0; target_idx < params.target_popnames.size(); ++target_idx) {
    const auto &pop = params.target_popnames[target_idx];
    std::vector<std::string> id_labels;
    id_labels.push_back(pop);
    for (const auto &entry : filtered_ids) {
      if (entry.population == pop) {
        id_labels.push_back(entry.individual);
      }
    }
    std::vector<int> matches;
    for (const auto &label : id_labels) {
      auto it = row_lookup.find(label);
      if (it != row_lookup.end()) {
        matches.insert(matches.end(), it->second.begin(), it->second.end());
      }
    }
    if (matches.empty()) {
      std::cerr << "NO INDS OF " << pop << " FOUND AMONG ROWS OF "
                << params.copyvector_file << "! Exiting...." << std::endl;
      return EXIT_FAILURE;
    }
    for (std::size_t col_idx = 0; col_idx < params.donor_pops_all.size(); ++col_idx) {
      double sum = 0.0;
      for (int row_idx : matches) {
        sum += predmat_orig[row_idx][col_idx];
      }
      recipient_mat[target_idx][col_idx] = sum / static_cast<double>(matches.size());
    }
  }

  std::ofstream(params.save_file, std::ios::trunc).close();

  std::random_device rd;
  std::mt19937_64 rng(rd());

  for (std::size_t i = 0; i < params.target_popnames.size(); ++i) {
    std::cout << "Analysing target " << (i + 1) << " of " << params.target_popnames.size()
              << " -- " << params.target_popnames[i] << "....." << std::endl;
    std::vector<double> target_vec = recipient_mat[i];
    std::vector<std::vector<double>> surr_mat = predmat;
    std::vector<std::string> row_names = donor_pops_all2;
    std::vector<std::string> surr_pops = donor_pops_all2;

    std::vector<int> row_indices;
    for (std::size_t r = 0; r < donor_pops_all2.size(); ++r) {
      if (donor_pops_all2[r] == params.target_popnames[i]) {
        row_indices.push_back(static_cast<int>(r));
      }
    }
    std::vector<int> col_indices;
    for (std::size_t c = 0; c < params.donor_pops_all.size(); ++c) {
      if (params.donor_pops_all[c] == params.target_popnames[i]) {
        col_indices.push_back(static_cast<int>(c));
      }
    }

    if ((params.self_copy_ind == 0 && !row_indices.empty()) ||
        (params.self_copy_ind == 1 && col_indices.empty() && !row_indices.empty())) {
      std::vector<std::vector<double>> new_surr_mat;
      std::vector<std::string> new_row_names;
      for (std::size_t r = 0; r < surr_mat.size(); ++r) {
        if (std::find(row_indices.begin(), row_indices.end(), static_cast<int>(r)) == row_indices.end()) {
          new_surr_mat.push_back(surr_mat[r]);
          new_row_names.push_back(row_names[r]);
        }
      }
      surr_mat = std::move(new_surr_mat);
      row_names = std::move(new_row_names);
    }

    if (params.self_copy_ind == 1 && !col_indices.empty()) {
      std::vector<double> self_copy_vec(params.donor_pops_all.size(), 0.0);
      for (int idx : col_indices) {
        self_copy_vec[idx] = 1.0;
      }
      if (!row_indices.empty()) {
        for (int idx : row_indices) {
          if (idx >= 0 && idx < static_cast<int>(surr_mat.size())) {
            surr_mat[idx] = self_copy_vec;
          }
        }
      } else {
        surr_mat.push_back(self_copy_vec);
        row_names.push_back(params.target_popnames[i]);
        surr_pops.push_back(params.target_popnames[i]);
      }
    }

    std::vector<double> row_sums(surr_mat.size(), 0.0);
    for (std::size_t r = 0; r < surr_mat.size(); ++r) {
      row_sums[r] = std::accumulate(surr_mat[r].begin(), surr_mat[r].end(), 0.0);
      if (row_sums[r] == 0.0) {
        throw std::runtime_error("Encountered zero row sum in surrogate matrix.");
      }
    }
    for (std::size_t r = 0; r < surr_mat.size(); ++r) {
      for (double &val : surr_mat[r]) {
        val /= row_sums[r];
      }
    }

    sourcefind(target_vec, surr_mat, params.num_pops, params.surr_exp, params.num_slots,
               params.num_runs, params.burn_in, params.thin_val, params.target_popnames[i],
               surr_pops, row_names, params.save_file, rng);
  }

  std::cout << "Finished!" << std::endl;
  return 0;
}

