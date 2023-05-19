import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import operator
import argparse
import sys
import matplotlib.pyplot as plt
plt.style.use('data/plots_paper.mplstyle')
import pathlib


class algs:
    def __init__(self, exp='genre', p=0.0, padval=0.0):

        assert(exp in ['genre', 'movie', 'book'])
        self.exp_name = exp
        self.p = p
        self.padval = padval

        #Load data
        self.tables = pd.read_pickle(f'preproc/{self.exp_name}s/{self.exp_name}_tables_train.pkl')
        self.test_data = pd.read_pickle(f'preproc/{self.exp_name}s/test_data_usercount')
        self.true_means_test = pd.read_pickle(f'preproc/{self.exp_name}s/true_means_test')

        self.numArms = len(self.tables.keys())
        self.optArm = np.argmax(self.true_means_test)

    def generate_sample(self, arm):

        d = self.test_data[self.test_data[f'{self.exp_name}_col'] == arm]
        reward = d['Rating'].sample(n=1, replace = True)

        return reward

    # The ThompsonSample function generates Thompson samples for each arm based on empirical means and number of pulls.
    def ThompsonSample(self, empiricalMean, numPulls, beta):
        numArms = self.numArms
        sampleArm = np.zeros(numArms)

        # Compute the standard deviation and variance for the normal distribution
        var_ = beta / (numPulls + 1.0)
        std_ = np.sqrt(var_)

        mean_ = empiricalMean

        # Generate samples from the normal distribution with the calculated mean and standard deviation
        sampleArm = np.random.normal(mean_, std_)

        return sampleArm

    # The UCB function implements the Upper Confidence Bound algorithm for multi-armed bandits.
    def UCB(self, num_iterations, T):
        numArms = self.numArms
        optArm = self.optArm
        true_means_test = self.true_means_test
        tables = self.tables

        # Initialize the exploration parameter B for all arms
        B = [5.0] * numArms

        # Initialize the average regret for UCB
        avg_ucb_regret = np.zeros((num_iterations, T))

        # Iterate through the number of iterations
        for iteration in tqdm(range(num_iterations)):
            # Initialize variables for UCB algorithm
            UCB_pulls = np.zeros(numArms)
            UCB_index = np.zeros(numArms)
            UCB_empReward = np.zeros(numArms)
            UCB_sumReward = np.zeros(numArms)

            UCB_index[:] = np.inf
            UCB_empReward[:] = np.inf

            ucb_regret = np.zeros(T)
            for t in range(T):
                # Pull each arm once at the beginning
                if t < numArms:
                    UCB_kt = t
                else:
                    # Select the arm with the highest UCB index
                    UCB_kt = np.argmax(UCB_index)

                # Generate the reward for the selected arm
                UCB_reward = self.generate_sample(UCB_kt)

                # Update the number of pulls and the sum of rewards for the selected arm
                UCB_pulls[UCB_kt] = UCB_pulls[UCB_kt] + 1
                UCB_sumReward[UCB_kt] = UCB_sumReward[UCB_kt] + float(UCB_reward.iloc[0])
                UCB_empReward[UCB_kt] = UCB_sumReward[UCB_kt] / float(UCB_pulls[UCB_kt])

                # Update the UCB index for all arms that have been pulled at least once
                for k in range(numArms):
                    if UCB_pulls[k] > 0:
                        UCB_index[k] = UCB_empReward[k] + B[k] * np.sqrt(2.0 * np.log(t + 1) / UCB_pulls[k])

                # Calculate the regret for the current time step
                if t == 0:
                    ucb_regret[t] = true_means_test[optArm] - true_means_test[UCB_kt]
                else:
                    ucb_regret[t] = ucb_regret[t - 1] + true_means_test[optArm] - true_means_test[UCB_kt]

            # Update the average regret for the current iteration
            avg_ucb_regret[iteration, :] = ucb_regret

        return avg_ucb_regret

    # The TS function implements the Thompson Sampling algorithm for multi-armed bandits.
    def TS(self, num_iterations, T):
        numArms = self.numArms
        optArm = self.optArm
        true_means_test = self.true_means_test
        tables = self.tables

        beta = 4.0

        # Initialize the average regret for Thompson Sampling
        avg_ts_regret = np.zeros((num_iterations, T))

        # Iterate through the number of iterations
        for iteration in tqdm(range(num_iterations)):
            numPulls = np.zeros(numArms)
            empReward = np.zeros(numArms)

            ts_regret = np.zeros(T)
            for t in range(T):
                # Pull each arm once at the beginning
                if t < numArms:
                    numPulls[t] += 1
                    assert numPulls[t] == 1

                    reward = self.generate_sample(t)
                    empReward[t] = float(reward.iloc[0])

                    if t != 0:
                        ts_regret[t] = ts_regret[t - 1] + true_means_test[optArm] - true_means_test[t]

                    continue

                # Generate Thompson samples for each arm and select the arm with the highest sample
                thompson = self.ThompsonSample(empReward, numPulls, beta)
                next_arm = np.argmax(thompson)

                # Generate reward, update pulls and empirical reward
                reward = self.generate_sample(next_arm)
                empReward[next_arm] = (empReward[next_arm] * numPulls[next_arm] + float(reward.iloc[0])) / (numPulls[next_arm] + 1)
                numPulls[next_arm] = numPulls[next_arm] + 1

                # Evaluate regret
                ts_regret[t] = ts_regret[t - 1] + true_means_test[optArm] - true_means_test[next_arm]

            # Update the average regret for the current iteration
            avg_ts_regret[iteration, :] = ts_regret

        return avg_ts_regret

    def C_UCB(self, num_iterations, T):
        numArms = self.numArms
        optArm = self.optArm
        true_means_test = self.true_means_test
        tables = self.tables

        B = [5.] * numArms

        # Initialize regret array
        avg_cucb_regret = np.zeros((num_iterations, T))

        # Run the algorithm for each iteration
        for iteration in tqdm(range(num_iterations)):
            pulls = np.zeros(numArms)
            empReward = np.zeros(numArms)
            sumReward = np.zeros(numArms)
            Index = dict(zip(range(numArms), [np.inf] * numArms))

            empReward[:] = np.inf

            # Initialize pseudo-reward arrays
            empPseudoReward = np.zeros((numArms, numArms))
            sumPseudoReward = np.zeros((numArms, numArms))
            empPseudoReward[:, :] = np.inf

            cucb_regret = np.zeros(T)
            for t in range(T):
                # Identify arms in the set \ell
                bool_ell = pulls >= (float(t - 1) / numArms)

                # Find the maximum and second maximum empirical rewards in the set \ell
                max_mu_hat = np.max(empReward[bool_ell])

                if empReward[bool_ell].shape[0] == 1:
                    secmax_mu_hat = max_mu_hat
                else:
                    temp = empReward[bool_ell]
                    temp[::-1].sort()
                    secmax_mu_hat = temp[1]
                argmax_mu_hat = np.where(empReward == max_mu_hat)[0][0]

                # Identify the competitive set of arms
                min_phi = np.min(empPseudoReward[:, bool_ell], axis=1)

                comp_set = set()
                # Add the argmax arm
                comp_set.add(argmax_mu_hat)

                # Add other competitive arms
                for arm in range(numArms):
                    if arm != argmax_mu_hat and min_phi[arm] >= max_mu_hat:
                        comp_set.add(arm)
                    elif arm == argmax_mu_hat and min_phi[arm] >= secmax_mu_hat:
                        comp_set.add(arm)

                # Select the next arm to pull
                if t < numArms:
                    k_t = t % numArms
                elif len(comp_set) == 0:
                    # UCB for an empty competitive set
                    k_t = max(Index.items(), key=operator.itemgetter(1))[0]
                else:
                    comp_Index = {ind: Index[ind] for ind in comp_set}
                    k_t = max(comp_Index.items(), key=operator.itemgetter(1))[0]

                # Update the number of pulls and rewards
                pulls[k_t] = pulls[k_t] + 1
                reward = self.generate_sample(k_t)

                sumReward[k_t] = sumReward[k_t] + float(reward.iloc[0])
                empReward[k_t] = sumReward[k_t] / float(pulls[k_t])

                # Update pseudo-rewards
                pseudoRewards = tables[k_t][reward - 1, :]
                sumPseudoReward[:, k_t] = sumPseudoReward[:, k_t] + pseudoRewards
                empPseudoReward[:, k_t] = np.divide(sumPseudoReward[:, k_t], float(pulls[k_t]))

                # Update diagonal elements of pseudo-rewards
                empPseudoReward[np.arange(numArms), np.arange(numArms)] = empReward

                # Update UCB+LCB indices using pseudo-rewards
                for k in range(numArms):
                    if pulls[k] > 0:
                        # UCB index
                        Index[k] = empReward[k] + B[k] * np.sqrt(2. * np.log(t + 1) / pulls[k])

                # Calculate regret
                if t == 0:
                    cucb_regret[t] = true_means_test[optArm] - true_means_test[k_t]
                else:
                    cucb_regret[t] = cucb_regret[t - 1] + true_means_test[optArm] - true_means_test[k_t]

            # Store regret for each iteration
            avg_cucb_regret[iteration, :] = cucb_regret

        return avg_cucb_regret

    def C_TS(self, num_iterations, T):
        
        numArms = self.numArms
        optArm = self.optArm
        true_means_test = self.true_means_test
        tables = self.tables
        
        B = [5.] * numArms

        # Initialize regret array
        avg_tsc_regret = np.zeros((num_iterations, T))

        beta = 4.  # since sigma was taken as 2

        # Run the algorithm for each iteration
        for iteration in tqdm(range(num_iterations)):
            TSC_pulls = np.zeros(numArms)

            TSC_empReward = np.zeros(numArms)
            TSC_sumReward = np.zeros(numArms)

            TSC_empReward[:] = np.inf

            # Initialize pseudo-reward arrays
            TSC_empPseudoReward = np.zeros((numArms, numArms))
            TSC_sumPseudoReward = np.zeros((numArms, numArms))

            TSC_empPseudoReward[:, :] = np.inf

            tsc_regret = np.zeros(T)

            for t in range(T):
                # Identify arms in the set \ell
                bool_ell = TSC_pulls >= (float(t - 1) / numArms)

                # Find the maximum and second maximum empirical rewards in the set \ell
                max_mu_hat = np.max(TSC_empReward[bool_ell])

                if TSC_empReward[bool_ell].shape[0] == 1:
                    secmax_mu_hat = max_mu_hat
                else:
                    temp = TSC_empReward[bool_ell]
                    temp[::-1].sort()
                    secmax_mu_hat = temp[1]
                argmax_mu_hat = np.where(TSC_empReward == max_mu_hat)[0][0]

                # Identify the competitive set of arms
                min_phi = np.min(TSC_empPseudoReward[:, bool_ell], axis=1)

                comp_set = set()
                # Add the argmax arm
                comp_set.add(argmax_mu_hat)

                # Add other competitive arms
                for arm in range(numArms):
                    if arm != argmax_mu_hat and min_phi[arm] >= max_mu_hat:
                        comp_set.add(arm)
                    elif arm == argmax_mu_hat and min_phi[arm] >= secmax_mu_hat:
                        comp_set.add(arm)

                if t < numArms:
                    k_t = t  # % numArms
                else:
                    # Thompson Sampling
                    thompson = self.ThompsonSample(TSC_empReward, TSC_pulls, beta)
                    comp_values = {ind: thompson[ind] for ind in comp_set}
                    k_t = max(comp_values.items(), key=operator.itemgetter(1))[0]

                # Update the number of pulls and rewards
                TSC_pulls[k_t] = TSC_pulls[k_t] + 1

                reward = self.generate_sample(k_t)

                # Update \mu_{k_t}
                TSC_sumReward[k_t] = TSC_sumReward[k_t] + float(reward.iloc[0])
                TSC_empReward[k_t] = TSC_sumReward[k_t] / float(TSC_pulls[k_t])

                # Pseudo-reward updates
                TSC_pseudoRewards = tables[k_t][reward - 1, :]  # (zero-indexed)

                TSC_sumPseudoReward[:, k_t] = TSC_sumPseudoReward[:, k_t] + TSC_pseudoRewards
                TSC_empPseudoReward[:, k_t] = np.divide(TSC_sumPseudoReward[:, k_t], float(TSC_pulls[k_t]))
                
                # Calculate regret
                if t == 0:
                    tsc_regret[t] = true_means_test[optArm] - true_means_test[k_t]
                else:
                    tsc_regret[t] = tsc_regret[t - 1] + true_means_test[optArm] - true_means_test[k_t]

            # Store regret for each iteration
            avg_tsc_regret[iteration, :] = tsc_regret

        return avg_tsc_regret

    def run(self, num_iterations=20, T=5000):

        avg_ucb_regret = self.UCB(num_iterations, T)
        avg_ts_regret = self.TS(num_iterations, T)
        avg_cucb_regret = self.C_UCB(num_iterations, T)
        avg_cts_regret = self.C_TS(num_iterations, T)

        # mean cumulative regret
        self.plot_av_ucb = np.mean(avg_ucb_regret, axis=0)
        self.plot_av_ts = np.mean(avg_ts_regret, axis=0)
        self.plot_av_cucb = np.mean(avg_cucb_regret, axis=0)
        self.plot_av_cts = np.mean(avg_cts_regret, axis=0)

        # std dev over runs
        self.plot_std_ucb = np.sqrt(np.var(avg_ucb_regret, axis=0))
        self.plot_std_ts = np.sqrt(np.var(avg_ts_regret, axis=0))
        self.plot_std_cucb = np.sqrt(np.var(avg_cucb_regret, axis=0))
        self.plot_std_cts = np.sqrt(np.var(avg_cts_regret, axis=0))

        self.save_data()

    def edit_data(self):

        if self.exp_name == 'genre':
            # code only masks values as done in the paper
            genre_tables = pd.read_pickle(f'preproc/{self.exp_name}s/genre_tables_train.pkl')
            p = self.p
            for genre in range(18):
                for row in range(genre_tables[genre].shape[0]):
                    row_len = int(genre_tables[genre].shape[1])
                    genre_tables[genre][row][np.random.choice(np.arange(row_len), size=int(p*row_len), replace=False)] = 5.
            # restore reference columns
            for genre in range(18):
                genre_tables[genre][:, genre] = np.arange(1,6)

            self.tables = genre_tables

        elif self.exp_name == 'movie':
            # code only pads entries as done in the paper
            movie_tables = pd.read_pickle(f'preproc/{self.exp_name}s/movie_tables_train.pkl')
            pad_val = self.padval
            for movie in range(50): # top 50 movies picked in preproc
                movie_tables[movie] += pad_val
                movie_tables[movie][movie_tables[movie] > 5] = 5.
                movie_tables[movie][:,movie] = np.arange(1,6)

            self.tables = movie_tables

        elif self.exp_name == 'book':
            book_tables = pd.read_pickle(f'preproc/{self.exp_name}s/book_tables_train.pkl')
            p = self.p
            pad_val = self.padval
            for book in range(25): # top 25 books picked in preproc
                for row in range(book_tables[book].shape[0]):
                    row_len = int(book_tables[book].shape[1])
                    book_tables[book][row][np.random.choice(np.arange(row_len), size= int(p*row_len),
                                                            replace=False)] = 5.

                book_tables[book] += pad_val
                book_tables[book][book_tables[book] > 5] = 5.

                book_tables[book][:, book] = np.arange(1,6)

            self.tables = book_tables

    def save_data(self):
        algorithms = ['ucb', 'ts', 'cucb', 'cts']
        pathlib.Path(f'plot_arrays/{self.exp_name}s/').mkdir(parents=False, exist_ok=True)
        for alg in algorithms:
            np.save(f'plot_arrays/{self.exp_name}s/plot_av_{alg}_p{self.p:.2f}_pad{self.padval:.2f}',
                    getattr(self, f'plot_av_{alg}'))
            np.save(f'plot_arrays/{self.exp_name}s/plot_std_{alg}_p{self.p:.2f}_pad{self.padval:.2f}',
                    getattr(self, f'plot_std_{alg}'))

    def plot(self):
        spacing = 400
        # Means
        plt.plot(range(0, 5000)[::spacing], self.plot_av_ucb[::spacing], label='UCB', color='red', marker='+')
        plt.plot(range(0, 5000)[::spacing], self.plot_av_ts[::spacing], label='TS', color='yellow', marker='o')
        plt.plot(range(0, 5000)[::spacing], self.plot_av_cucb[::spacing], label='C-UCB', color='blue', marker='^')
        plt.plot(range(0, 5000)[::spacing], self.plot_av_cts[::spacing], label='C-TS', color='black', marker='x')
        # Confidence bounds
        plt.fill_between(range(0, 5000)[::spacing], (self.plot_av_ucb + self.plot_std_ucb)[::spacing],
                         (self.plot_av_ucb - self.plot_std_ucb)[::spacing], alpha=0.3, facecolor='r')
        plt.fill_between(range(0, 5000)[::spacing], (self.plot_av_ts + self.plot_std_ts)[::spacing],
                         (self.plot_av_ts - self.plot_std_ts)[::spacing], alpha=0.3, facecolor='y')
        plt.fill_between(range(0, 5000)[::spacing], (self.plot_av_cucb + self.plot_std_cucb)[::spacing],
                         (self.plot_av_cucb - self.plot_std_cucb)[::spacing], alpha=0.3, facecolor='b')
        plt.fill_between(range(0, 5000)[::spacing], (self.plot_av_cts + self.plot_std_cts)[::spacing],
                         (self.plot_av_cts - self.plot_std_cts)[::spacing], alpha=0.3, facecolor='k')
        # Plot
        plt.legend()
        plt.grid(True, axis='y')
        plt.xlabel('Number of Rounds')
        plt.ylabel('Cumulative Regret')
        # Save
        pathlib.Path('data/plots/').mkdir(parents=False, exist_ok=True)
        plt.savefig(f'data/plots/{self.exp_name}_p{self.p:.2f}_pad{self.padval:.2f}.pdf')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', dest='exp', type=str, default='genre', help="Experiment to run (genre, movie, book)")
    parser.add_argument('--num_iterations', dest='num_iterations', type=int, default=20,
                        help="Number of iterations of each run")
    parser.add_argument('--T', dest='T', type=int, default=5000, help="Number of rounds")
    parser.add_argument('--p', dest='p', type=float, default=0.0, help="Fraction of table entries to mask")
    parser.add_argument('--padval', dest='padval', type=float, default=0.0, help="Padding value for table entries")
    return parser.parse_args()


def main(args):
    args = parse_arguments()
    bandit_obj = algs(args.exp, p=args.p, padval=args.padval)
    bandit_obj.edit_data()
    bandit_obj.run(args.num_iterations, args.T)
    bandit_obj.plot()


if __name__ == '__main__':
    main(sys.argv)
