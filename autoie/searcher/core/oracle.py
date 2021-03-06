# Copyright 2019 The Keras Tuner Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Oracle base class."""

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import glob
import collections
import json
import logging

from autoie.searcher.core import trial as trial_lib, hyperparameters as hp_module
from autoie.searcher.core.trial import Stateful
from autoie.utils import metric
from autoie.utils.common import create_directory

Objective = collections.namedtuple('Objective', 'name direction')
LOGGER = logging.getLogger(__name__)


class Oracle(Stateful):
    """Implements a hyperparameter optimization algorithm.

    Attributes:
        objective: String. Name of model metric to minimize
            or maximize, e.g. "val_accuracy".
        max_trials: The maximum number of hyperparameter
            combinations to try.
        hyperparameters: HyperParameters class instance.
            Can be used to override (or register in advance)
            hyperparamters in the search space.
        tune_new_entries: Whether hyperparameter entries
            that are requested by the hypermodel
            but that were not specified in `hyperparameters`
            should be added to the search space, or not.
            If not, then the default value for these parameters
            will be used.
        allow_new_entries: Whether the hypermodel is allowed
            to request hyperparameter entries not listed in
            `hyperparameters`.
    """

    def __init__(self,
                 objective,
                 max_trials=None,
                 hyperparameters=None,
                 allow_new_entries=True,
                 tune_new_entries=True):
        self.objective = _format_objective(objective)
        self.max_trials = max_trials
        if not hyperparameters:
            if not tune_new_entries:
                raise ValueError(
                    'If you set `tune_new_entries=False`, you must'
                    'specify the search space via the '
                    '`hyperparameters` argument.')
            if not allow_new_entries:
                raise ValueError(
                    'If you set `allow_new_entries=False`, you must'
                    'specify the search space via the '
                    '`hyperparameters` argument.')
            self.hyperparameters = hp_module.HyperParameters()
        else:
            self.hyperparameters = hyperparameters
        self.allow_new_entries = allow_new_entries
        self.tune_new_entries = tune_new_entries

        # trial_id -> Trial
        self.trials = {}
        # tuner_id -> Trial
        self.ongoing_trials = {}

        # Set in `BaseTuner` via `set_project_dir`.
        self._directory = None
        self._project_name = None

    def _populate_space(self, trial_id):
        """Fill the hyperparameter space with values for a trial.

        This method should be overrridden in subclasses and called in
        `create_trial` in order to populate the hyperparameter space with
        values.

        Args:
          `trial_id`: The id for this Trial.

        Returns:
            A dictionary with keys "values" and "status", where "values" is
            a mapping of parameter names to suggested values, and "status"
            is the TrialStatus that should be returned for this trial (one
            of "RUNNING", "IDLE", or "STOPPED").
        """
        raise NotImplementedError

    def _score_trial(self, trial):
        """Score a completed `Trial`.

        This method can be overridden in subclasses to provide a score for
        a set of hyperparameter values. This method is called from `end_trial`
        on completed `Trial`s.

        Args:
          trial: A completed `Trial` object.
        """
        # Assumes single objective, subclasses can override.
        trial.score = trial.metrics.get_best_value(self.objective.name)
        trial.best_step = trial.metrics.get_best_step(self.objective.name)

    def create_trial(self, tuner_id):
        """Create a new `Trial` to be run by the `Tuner`.

        A `Trial` corresponds to a unique set of hyperparameters to be run
        by `Tuner.run_trial`.

        Args:
          tuner_id: A ID that identifies the `Tuner` requesting a
          `Trial`. `Tuners` that should run the same trial (for instance,
           when running a multi-worker model) should have the same ID.

        Returns:
          A `Trial` object containing a set of hyperparameter values to run
          in a `Tuner`.
        """
        # Allow for multi-worker DistributionStrategy within a Trial.
        if tuner_id in self.ongoing_trials:
            return self.ongoing_trials[tuner_id]

        trial_id = trial_lib.generate_trial_id()

        if len(self.trials) >= self.max_trials:
            status = trial_lib.TrialStatus.STOPPED
            values = None
        else:
            response = self._populate_space(trial_id)
            status = response['status']
            values = response['values'] if 'values' in response else None

        hyperparameters = self.hyperparameters.copy()
        hyperparameters.values = values or {}
        trial = trial_lib.Trial(
            hyperparameters=hyperparameters,
            trial_id=trial_id,
            status=status)

        if status == trial_lib.TrialStatus.RUNNING:
            self.ongoing_trials[tuner_id] = trial
            self.trials[trial_id] = trial
            self._save_trial(trial)
            self.save()

        return trial

    def update_trial(self, trial_id, metrics, step=0):
        """Used by a worker to report the status of a trial.

        Args:
            trial_id: A previously seen trial id.
            metrics: Dict of float. The current value of this
                trial's metrics.
            step: (Optional) Float. Used to report intermediate results. The
                current value in a timeseries representing the state of the
                trial. This is the value that `metrics` will be associated with.

        Returns:
            Trial object. Trial.status will be set to "STOPPED" if the Trial
            should be stopped early.
        """
        trial = self.trials[trial_id]
        self._check_objective_found(metrics)
        for metric_name, metric_value in metrics.items():
            if not trial.metrics.exists(metric_name):
                direction = _maybe_infer_direction_from_objective(
                    self.objective, metric_name)
                trial.metrics.register(metric_name, direction=direction)
            trial.metrics.update(metric_name, metric_value, step=step)
        self._save_trial(trial)
        # To signal early stopping, set Trial.status to "STOPPED".
        return trial.status

    def end_trial(self, trial_id, status='COMPLETED'):
        """Record the measured objective for a set of parameter values.

        Args:
            trial_id: String. Unique id for this trial.
            status: String, one of "COMPLETED", "INVALID". A status of
                "INVALID" means a trial has crashed or been deemed
                infeasible.
        """
        trial = None
        for tuner_id, ongoing_trial in self.ongoing_trials.items():
            if ongoing_trial.trial_id == trial_id:
                trial = self.ongoing_trials.pop(tuner_id)
                break

        if not trial:
            raise ValueError(
                'Ongoing trial with id: {} not found.'.format(trial_id))

        trial.status = status
        if status == trial_lib.TrialStatus.COMPLETED:
            self._score_trial(trial)
        self._save_trial(trial)
        self.save()

    def get_space(self):
        """Returns the `HyperParameters` search space."""
        return self.hyperparameters.copy()

    def update_space(self, hyperparameters):
        """Add new hyperparameters to the tracking space.

        Already recorded parameters get ignored.

        Args:
            hyperparameters: An updated HyperParameters object.
        """
        ref_names = {hp.name for hp in self.hyperparameters.space}
        new_hps = [hp for hp in hyperparameters.space
                   if hp.name not in ref_names]

        if new_hps and not self.allow_new_entries:
            raise RuntimeError('`allow_new_entries` is `False`, but found '
                               'new entries {}'.format(new_hps))

        if not self.tune_new_entries:
            # New entries should always use the default value.
            return

        for hp in new_hps:
            self.hyperparameters.register(
                hp.name, hp.__class__.__name__, hp.get_config())

    def get_trial(self, trial_id):
        """Returns the `Trial` specified by `trial_id`."""
        return self.trials[trial_id]

    def get_best_trials(self, num_trials=1):
        """Returns the best `Trial`s."""
        trials = [t for t in self.trials.values()
                  if t.status == trial_lib.TrialStatus.COMPLETED]

        sorted_trials = sorted(
            trials,
            key=lambda trial: trial.score,
            # Assumes single objective, subclasses can override.
            reverse=self.objective.direction == 'max'
        )
        return sorted_trials[:num_trials]

    def remaining_trials(self):
        if self.max_trials:
            return self.max_trials - len(self.trials.items())
        else:
            return None

    def get_state(self):
        # `self.trials` are saved in their own, Oracle-agnostic files.
        # Just save the IDs for ongoing trials, since these are in `trials`.
        state = {}
        state['ongoing_trials'] = {
            tuner_id: trial.trial_id
            for tuner_id, trial in self.ongoing_trials.items()}
        # Hyperparameters are part of the state because they can be added to
        # during the course of the search.
        state['hyperparameters'] = self.hyperparameters.get_config()
        return state

    def set_state(self, state):
        # `self.trials` are saved in their own, Oracle-agnostic files.
        self.ongoing_trials = {
            tuner_id: self.trials[trial_id]
            for tuner_id, trial_id in state['ongoing_trials'].items()}
        self.hyperparameters = hp_module.HyperParameters.from_config(
            state['hyperparameters'])

    def set_project_dir(self, directory, project_name, overwrite=False):
        """Sets the project directory and reloads the Oracle."""
        self._directory = directory
        self._project_name = project_name
        if (not overwrite) and os.path.exists(self._get_oracle_fname()):
            LOGGER.info('Reloading Oracle from {}'.format(
                self._get_oracle_fname()))
            self.reload()

    @property
    def _project_dir(self):
        dirname = os.path.join(
            self._directory,
            self._project_name)
        create_directory(dirname)
        return dirname

    def save(self):
        # `self.trials` are saved in their own, Oracle-agnostic files.
        super(Oracle, self).save(self._get_oracle_fname())

    def reload(self):
        # Reload trials from their own files.
        trial_dirs = glob.glob(os.path.join(self._project_dir, 'trial_*'))
        trial_fnames = [os.path.join(trial_dir, 'trial.json') for trial_dir in trial_dirs]
        for fname in trial_fnames:
            with open(fname, 'r') as fp:
                trial_state = json.load(fp)
            trial = trial_lib.Trial.from_state(trial_state)
            self.trials[trial.trial_id] = trial
        super(Oracle, self).reload(self._get_oracle_fname())

    def _get_oracle_fname(self):
        return os.path.join(
            self._project_dir,
            'oracle.json')

    @staticmethod
    def _compute_values_hash(values):
        keys = sorted(values.keys())
        s = ''.join(str(k) + '=' + str(values[k]) for k in keys)
        # return hashlib.sha256(s.encode('utf-8')).hexdigest()[:32]
        return hash(s)

    def _check_objective_found(self, metrics):
        if isinstance(self.objective, Objective):
            objective_names = [self.objective.name]
        else:
            objective_names = [obj.name for obj in self.objective]
        for metric_name in metrics.keys():
            if metric_name in objective_names:
                objective_names.remove(metric_name)
        if objective_names:
            raise ValueError(
                'Objective value missing in metrics reported to the '
                'Oracle, expected: {}, found: {}'.format(
                    objective_names, metrics.keys()))

    def _get_trial_dir(self, trial_id):
        dirname = os.path.join(
            self._project_dir,
            'trial_' + str(trial_id))
        create_directory(dirname)
        return dirname

    def _save_trial(self, trial):
        # Write trial status to trial directory
        trial_id = trial.trial_id
        trial.save(os.path.join(
            self._get_trial_dir(trial_id),
            'trial.json'))


def _format_objective(objective):
    if isinstance(objective, list):
        return [_format_objective(obj) for obj in objective]
    if isinstance(objective, Objective):
        return objective
    if isinstance(objective, str):
        direction = metric.infer_metric_direction(objective)
        return Objective(name=objective, direction=direction)
    else:
        raise ValueError('`objective` not understood, expected str or '
                         '`Objective` object, found: {}'.format(objective))


def _maybe_infer_direction_from_objective(objective, metric_name):
    if isinstance(objective, Objective):
        objective = [objective]
    for obj in objective:
        if obj.name == metric_name:
            return obj.direction
    return None
