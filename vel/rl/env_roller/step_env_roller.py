import torch
import numpy as np

from vel.api import BatchInfo, Model
from vel.rl.api import Trajectories, Rollout, EnvRollerBase, EnvRollerFactoryBase
from vel.util.tensor_accumulator import TensorAccumulator
from bc_gym_planning_env.envs.base.action import Action


class StepEnvRoller(EnvRollerBase):
    """
    Class calculating env rollouts.
    """

    def __init__(self, environment, device):
        self._environment = environment
        self.device = device

        # Initial observation - kept on CPU
        # self.last_observation = torch.from_numpy(self.environment.reset()).clone()
        self.last_observation = self._bc_observations_to_tensor(self.environment.reset()).clone()

        # Relevant for RNN policies - kept on DEVICE
        self.hidden_state = None

    def _bc_observations_to_tensor(self, observations):
        """ Convert numpy array to a tensor """
        if isinstance(observations, dict):
            input1 = observations['environment']
            input2 = observations['goal']
            input_additional_channel = np.zeros(input1.shape[:-1] + (1,))
            input_additional_channel[:, 0, 0:5, :] = input2
            input = np.concatenate((input1.astype(float), input_additional_channel), axis=3)
            return torch.from_numpy(input).to(self.device)
        else:
            raise NotImplementedError

    @property
    def environment(self):
        """ Return environment of this env roller """
        return self._environment

    @torch.no_grad()
    def rollout(self, batch_info: BatchInfo, model: Model, number_of_steps: int) -> Rollout:
        """ Calculate env rollout """
        accumulator = TensorAccumulator()
        episode_information = []  # List of dictionaries with episode information

        if self.hidden_state is None and model.is_recurrent:
            self.hidden_state = model.zero_state(self.last_observation.size(0)).to(self.device)

        # Remember rollout initial state, we'll use that for training as well
        initial_hidden_state = self.hidden_state

        for step_idx in range(number_of_steps):
            if model.is_recurrent:
                step = model.step(self.last_observation.to(self.device), state=self.hidden_state)
                self.hidden_state = step['state']
            else:
                step = model.step(self.last_observation.to(self.device))

            # Add step to the tensor accumulator
            for name, tensor in step.items():
                accumulator.add(name, tensor.cpu())

            accumulator.add('observations', self.last_observation)

            actions_numpy = step['actions'].detach().cpu().numpy()
            # new_obs, new_rewards, new_dones, new_infos = self.environment.step(actions_numpy)
            action_classes = []
            for i in range(actions_numpy.shape[0]):
                action_class = Action(command=actions_numpy[i])
                action_classes.append(action_class)
            new_obs, new_rewards, new_dones, new_infos = self.environment.step(action_classes)

            # Done is flagged true when the episode has ended AND the frame we see is already a first frame from the
            # next episode
            dones_tensor = torch.from_numpy(new_dones.astype(np.float32)).clone()
            accumulator.add('dones', dones_tensor)

            # self.last_observation = torch.from_numpy(new_obs).clone()
            self.last_observation = self._bc_observations_to_tensor(new_obs).clone()

            if model.is_recurrent:
                # Zero out state in environments that have finished
                self.hidden_state = self.hidden_state * (1.0 - dones_tensor.unsqueeze(-1)).to(self.device)

            accumulator.add('rewards', torch.from_numpy(new_rewards.astype(np.float32)).clone())

            episode_information.append(new_infos)

        if model.is_recurrent:
            final_values = model.value(self.last_observation.to(self.device), state=self.hidden_state).cpu()
        else:
            final_values = model.value(self.last_observation.to(self.device)).cpu()

        accumulated_tensors = accumulator.result()

        return Trajectories(
            num_steps=accumulated_tensors['observations'].size(0),
            num_envs=accumulated_tensors['observations'].size(1),
            environment_information=episode_information,
            transition_tensors=accumulated_tensors,
            rollout_tensors={
                'initial_hidden_state': initial_hidden_state.cpu() if initial_hidden_state is not None else None,
                'final_values': final_values
            }
        )


class StepEnvRollerFactory(EnvRollerFactoryBase):
    """ Factory for the StepEnvRoller """
    def __init__(self):
        pass

    def instantiate(self, environment, device):
        return StepEnvRoller(
            environment=environment,
            device=device,
        )


def create():
    """ Vel factory function """
    return StepEnvRollerFactory()
