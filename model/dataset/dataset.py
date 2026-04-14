from torch.utils import data
import numpy as np
import torch
from .preprocessing import get_node_timestep_data

DEBUG_PRINTED = False


class EnvironmentDataset(object):
    def __init__(self, env, state, pred_state, node_freq_mult, scene_freq_mult, hyperparams, **kwargs):
        self.env = env
        self.state = state
        self.pred_state = pred_state
        self.hyperparams = hyperparams
        self.max_ht = self.hyperparams['maximum_history_length']
        self.max_ft = kwargs['min_future_timesteps']
        self.node_type_datasets = list()
        self._augment = False

        for node_type in env.NodeType:
            if node_type not in hyperparams['pred_state']:
                continue
            self.node_type_datasets.append(
                NodeTypeDataset(env, node_type, state, pred_state,
                                node_freq_mult, scene_freq_mult,
                                hyperparams, **kwargs)
            )

    @property
    def augment(self):
        return self._augment

    @augment.setter
    def augment(self, value):
        self._augment = value
        for node_type_dataset in self.node_type_datasets:
            node_type_dataset.augment = value

    def __iter__(self):
        return iter(self.node_type_datasets)


class NodeTypeDataset(data.Dataset):

    def __init__(self, env, node_type, state, pred_state,
                 node_freq_mult, scene_freq_mult, hyperparams,
                 augment=False, **kwargs):

        self.env = env
        self.state = state
        self.pred_state = pred_state
        self.hyperparams = hyperparams
        self.max_ht = self.hyperparams['maximum_history_length']
        self.max_ft = kwargs['min_future_timesteps']

        self.augment = augment

        self.node_type = node_type
        self.index = self.index_env(node_freq_mult, scene_freq_mult, **kwargs)
        self.len = len(self.index)
        self.edge_types = [
            edge_type for edge_type in env.get_edge_types()
            if edge_type[0] is node_type
        ]

    def apply_failures(self, data):
        global DEBUG_PRINTED

        mode = self.hyperparams.get("experiment_mode", "clean_clean")
        is_eval = self.hyperparams.get("is_eval", False)

        # 🧠 CONTROL LOGIC (NO MORE HACKS)

        # clean → clean
        if mode == "clean_clean":
            return data

        # clean → noisy (only eval noisy)
        if mode == "clean_noisy" and not is_eval:
            return data

        # noisy → clean (only eval clean)
        if mode == "noisy_clean" and is_eval:
            return data

        # noisy → noisy → always apply noise
        # (no return here → fall through)

        noise_params = self.hyperparams.get("noise_params", {})
        missing_prob = noise_params.get("missing_prob", 0.3)
        jitter_std = noise_params.get("jitter_std", 0.05)

        data = list(data)

        # 🔴 DEBUG (ONLY ONCE)
        if not DEBUG_PRINTED:
            print(f"[DEBUG] MODE: {mode} | is_eval: {is_eval}")
            print("🔥 NOISE ACTIVE")
            for i in range(len(data)):
                if isinstance(data[i], (np.ndarray, torch.Tensor)):
                    print(f"Index {i}: shape {data[i].shape}")
            DEBUG_PRINTED = True

        # 🔴 APPLY NOISE
        for i in range(len(data)):
            if isinstance(data[i], (np.ndarray, torch.Tensor)):

                if len(data[i].shape) == 2 and data[i].shape[1] == 2:

                    traj = data[i]

                    is_tensor = isinstance(traj, torch.Tensor)
                    if is_tensor:
                        traj = traj.cpu().numpy()

                    # Missing frames
                    if np.random.rand() < missing_prob:
                        mask = (np.random.rand(traj.shape[0]) > missing_prob).astype(float)
                        traj = traj * mask[:, None]

                    # Jitter
                    noise = np.random.normal(0, jitter_std, traj.shape)
                    traj = traj + noise

                    if is_tensor:
                        traj = torch.tensor(traj, dtype=data[i].dtype)

                    data[i] = traj

        return tuple(data)

    def index_env(self, node_freq_mult, scene_freq_mult, **kwargs):
        index = list()
        for scene in self.env.scenes:
            present_node_dict = scene.present_nodes(
                np.arange(0, scene.timesteps),
                type=self.node_type,
                **kwargs
            )
            for t, nodes in present_node_dict.items():
                for node in nodes:
                    index += [(scene, t, node)] * \
                             (scene.frequency_multiplier if scene_freq_mult else 1) * \
                             (node.frequency_multiplier if node_freq_mult else 1)
        return index

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        (scene, t, node) = self.index[i]

        if self.augment:
            scene = scene.augment()
            node = scene.get_node_by_id(node.id)

        data = get_node_timestep_data(
            self.env, scene, t, node,
            self.state, self.pred_state,
            self.edge_types,
            self.max_ht, self.max_ft,
            self.hyperparams
        )

        data = self.apply_failures(data)

        return data