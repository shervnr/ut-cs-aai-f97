# Final-Project
# Shervin Rasoulzadeh 610397039

from math import log10
import string
import sys

# a set of states
letters = list(string.ascii_lowercase)
letters.append('_')


def read_file(index):
    f = open(sys.argv[index], 'r')
    states_observations_seq = []
    for line in f:
        line_letters = line.split()
        states_observations_seq.append((line_letters[0], line_letters[1]))
    f.close()
    return states_observations_seq


def generate_states(states_observations_seq_train):
    state_dict = dict()

    for letter in letters:
        state_dict[letter] = {"total": 0}

    for i in range(0, len(states_observations_seq_train) - 1):
        if states_observations_seq_train[i + 1][0] in state_dict[states_observations_seq_train[i][0]]:
            state_dict[states_observations_seq_train[i][0]][states_observations_seq_train[i + 1][0]] += 1
        else:
            state_dict[states_observations_seq_train[i][0]][states_observations_seq_train[i + 1][0]] = 1
        state_dict[states_observations_seq_train[i][0]]["total"] += 1

    for i in letters:
        for j in letters:
            if j not in state_dict[i]:
                state_dict[i][j] = 0

    return state_dict


def generate_observations(states_observations):
    observation_dict = dict()

    for letter in letters:
        observation_dict[letter] = {"total": 0}

    for i in range(0, len(states_observations) - 1):
        if states_observations[i][1] in observation_dict[states_observations[i][0]]:
            observation_dict[states_observations[i][0]][states_observations[i][1]] += 1
        else:
            observation_dict[states_observations[i][0]][states_observations[i][1]] = 1
        observation_dict[states_observations[i][0]]["total"] += 1

    for i in letters:
        for j in letters:
            if j not in observation_dict[i]:
                observation_dict[i][j] = 0

    return observation_dict


def generate_initial_probability_distribution(states_dict, states_observations_seq_train):
    initial_probability = {}
    for letter in letters:
        initial_probability[letter] = states_dict[letter]["total"] / len(states_observations_seq_train)
    return initial_probability


def generate_transition_probabilities(states_dict):
    transition_probabilities = {}
    for i in letters:
        probability = []
        for j in letters:
            if j in states_dict[i]:
                temp = dict()
                # advantage of Laplace smoothing is that it avoids estimating any probabilities to be zero,
                #   even for events never observed in the data.
                # For HMMs, this is important since zero probabilities can be problematic for some algorithms.
                temp[j] = (states_dict[i][j] + 1) / (states_dict[i]['total'] + 27)
                probability.append(temp)
                transition_probabilities[i] = probability
    return transition_probabilities


def generate_emission_probabilities(observations_dict):
    emission_probabilities = {}
    for i in letters:
        probability = []
        for j in letters:
            if j in observations_dict[i]:
                temp = dict()
                # advantage of Laplace smoothing is that it avoids estimating any probabilities to be zero,
                #   even for events never observed in the data.
                # For HMMs, this is important since zero probabilities can be problematic for some algorithms.
                temp[j] = (observations_dict[i][j] + 1) / (observations_dict[i]["total"] + 27)
                probability.append(temp)
                emission_probabilities[i] = probability
    return emission_probabilities


def change_states_dict(states_dict):
    for i in states_dict:
        for j in states_dict[i]:
            if j != 'total':
                states_dict[i][j] = states_dict[i][j] / states_dict[i]["total"]
    return states_dict


def fix_transition_probabilities(transition_probabilities, states_dict_changed):
    for i in letters:
        for j in letters:
            for k in range(0, 27):
                if j in transition_probabilities[i][k]:
                    if states_dict_changed[i][j] != transition_probabilities[i][k][j]:
                        states_dict_changed[i][j] = transition_probabilities[i][k][j]
    return states_dict_changed


def change_observation_dict(observations_dict):
    for i in observations_dict:
        for j in observations_dict[i]:
            if j != 'total':
                observations_dict[i][j] = observations_dict[i][j] / observations_dict[i]["total"]
    return observations_dict


def fix_emissions_probabilities(emission_probabilities, observations_dict_changed):
    for i in letters:
        for j in letters:
            for k in range(0, 27):
                if j in emission_probabilities[i][k]:
                    if observations_dict_changed[i][j] != emission_probabilities[i][k][j]:
                        observations_dict_changed[i][j] = emission_probabilities[i][k][j]
    return observations_dict_changed


def viterbi(states_observations_test, initial_probability_distribution, new_observations, new_states):
    v = [{}]
    path = {}

    observations_sequence = []
    for i in range(0, len(states_observations_test) - 1):
        observations_sequence.append(states_observations_test[i][1])

    for i in letters:
        v[0][i] = log10(initial_probability_distribution[i]) + log10(new_observations[i][observations_sequence[0]])
        path[i] = [i]

    for t in range(1, len(observations_sequence)):
        v.append({})
        new_path = {}
        for j in letters:
            (prob, state) = max(
                (v[t - 1][i] + log10(new_states[i][j]) + log10(new_observations[j][observations_sequence[t]]), i)
                for i in letters)
            v[t][j] = prob
            new_path[j] = path[state] + [j]
        path = new_path

    # termination step
    (best_path_prob, best_path_pointer) = max((v[len(observations_sequence) - 1][s], s) for s in letters)
    return best_path_prob, path[best_path_pointer]


def write_file(best_path):
    f = open('FinalProject.txt', 'w')
    for letter in best_path:
        f.write(letter)
        f.write('\n')
    f.close()


def compute_error_rate(best_path, states_observations_seq_test):
    observations_path_corrects = 0.0
    viterbi_path_corrects = 0.0
    for i in range(len(best_path)):
        if states_observations_seq_test[i][0] == states_observations_seq_test[i][1]:
            observations_path_corrects += 1
        if best_path[i] == states_observations_seq_test[i][0]:
            viterbi_path_corrects += 1
    observation_path_error_rate = 1 - float((observations_path_corrects / len(best_path)))
    viterbi_path_error_rate = 1 - float((viterbi_path_corrects / len(best_path)))
    return observation_path_error_rate, viterbi_path_error_rate


def main():
    states_observations_seq_train = read_file(index=1)
    states_dict = generate_states(states_observations_seq_train)
    observations_dict = generate_observations(states_observations_seq_train)
    # an initial probability distribution over states. \pi_{i} is the probability that
    #   the Markov chain will start in state i
    initial_probability_distribution = generate_initial_probability_distribution(states_dict,
                                                                                 states_observations_seq_train)
    # each a_{i,j} representing the probability of moving from state i to state j,
    #   s.t \sum_{j=1}^{N} a_{i,j} = 1
    transition_probabilities = generate_transition_probabilities(states_dict)
    # a sequence of observation likelihoods, also called emission probabilities,
    #   each expressing the probability of an observation o_{t} being generated from a state i
    emission_probabilities = generate_emission_probabilities(observations_dict)
    states_dict_changed = change_states_dict(states_dict)
    new_states = fix_transition_probabilities(transition_probabilities, states_dict_changed)
    observations_dict_changed = change_observation_dict(observations_dict)
    new_observations = fix_transition_probabilities(emission_probabilities, observations_dict_changed)
    states_observations_seq_test = read_file(index=2)
    # Decoding : The Viterbi Algorithm
    # Viterbi algorithm for finding optimal sequence of hidden states. Given an observation sequence
    #   and an HMM \lambda = (A,B), the algorithm returns the state path through the HMM that assigns maximum likelihood
    #   to the observation sequence
    path_prob, best_path = viterbi(states_observations_seq_test, initial_probability_distribution, new_observations,
                                   new_states)
    write_file(best_path)
    error = compute_error_rate(best_path, states_observations_seq_test)
    print('starting error rate: ', end="")
    print(round(error[0] * 100, 1))
    print('ending error rate: ', end="")
    print(round(error[1] * 100, 1))


if __name__ == '__main__':
    main()
