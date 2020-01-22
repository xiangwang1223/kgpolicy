import collections
import numpy as np
import networkx as nx
from tqdm import tqdm


class CFData(object):
    def __init__(self, args_config):
        self.args_config = args_config

        path = args_config.data_path + args_config.dataset
        train_file = path + "/train.dat"
        test_file = path + "/test.dat"

        # ----------get number of users and items & then load rating data from train_file & test_file------------
        self.train_data = self._generate_interactions(train_file)
        self.test_data = self._generate_interactions(test_file)

        self.train_user_dict, self.test_user_dict = self._generate_user_dict()

        self.exist_users = list(self.train_user_dict.keys())
        self._statistic_interactions()

    # reading train & test interaction data.
    @staticmethod
    def _generate_interactions(file_name):
        inter_mat = list()

        lines = open(file_name, "r").readlines()
        for l in lines:
            tmps = l.strip()
            inters = [int(i) for i in tmps.split(" ")]

            u_id, pos_ids = inters[0], inters[1:]
            pos_ids = list(set(pos_ids))

            for i_id in pos_ids:
                inter_mat.append([u_id, i_id])

        return np.array(inter_mat)

    # generating user interaction dictionary.
    def _generate_user_dict(self):
        def _generate_dict(inter_mat):
            user_dict = dict()
            for u_id, i_id in inter_mat:
                if u_id not in user_dict.keys():
                    user_dict[u_id] = list()
                user_dict[u_id].append(i_id)
            return user_dict

        n_users = max(max(self.train_data[:, 0]), max(self.test_data[:, 0])) + 1

        # remap item id from [0, #items) to [#users, #users + #items).
        self.train_data[:, 1] = self.train_data[:, 1] + n_users
        self.test_data[:, 1] = self.test_data[:, 1] + n_users

        train_user_dict = _generate_dict(self.train_data)
        test_user_dict = _generate_dict(self.test_data)
        return train_user_dict, test_user_dict

    def _statistic_interactions(self):
        def _id_range(train_mat, test_mat, idx):
            min_id = min(min(train_mat[:, idx]), min(test_mat[:, idx]))
            max_id = max(max(train_mat[:, idx]), max(test_mat[:, idx]))
            n_id = max_id - min_id + 1
            return (min_id, max_id), n_id

        self.user_range, self.n_users = _id_range(
            self.train_data, self.test_data, idx=0
        )
        self.item_range, self.n_items = _id_range(
            self.train_data, self.test_data, idx=1
        )
        self.n_train = len(self.train_data)
        self.n_test = len(self.test_data)

        print("-" * 50)
        print("-     user_range: (%d, %d)" % (self.user_range[0], self.user_range[1]))
        print("-     item_range: (%d, %d)" % (self.item_range[0], self.item_range[1]))
        print("-        n_train: %d" % self.n_train)
        print("-         n_test: %d" % self.n_test)
        print("-        n_users: %d" % self.n_users)
        print("-        n_items: %d" % self.n_items)
        print("-" * 50)


class KGData(object):
    def __init__(self, args_config, entity_start_id=0, relation_start_id=0):
        self.args_config = args_config
        self.entity_start_id = entity_start_id
        self.relation_start_id = relation_start_id

        path = args_config.data_path + args_config.dataset
        kg_file = path + "/kg_final.txt"

        # ----------get number of entities and relations & then load kg data from kg_file ------------.
        self.kg_data, self.kg_dict, self.relation_dict = self._load_kg(kg_file)
        self._statistic_kg_triples()

    # reading train & test interaction data.
    def _load_kg(self, file_name):
        def _remap_kg_id(org_kg_np):
            new_kg_np = org_kg_np.copy()
            # consider the number of users
            new_kg_np[:, 0] = org_kg_np[:, 0] + self.entity_start_id
            new_kg_np[:, 2] = org_kg_np[:, 2] + self.entity_start_id
            # consider two additional relations --- 'interact' and 'be_interacted_with'.
            new_kg_np[:, 1] = org_kg_np[:, 1] + self.relation_start_id
            return new_kg_np

        def _construct_kg(kg_np):
            kg = collections.defaultdict(list)
            rd = collections.defaultdict(list)

            for head, relation, tail in kg_np:
                kg[head].append((tail, relation))
                rd[relation].append((head, tail))
            return kg, rd

        # get triplets with canonical direction like <item, has-aspect, entity>
        can_kg_np = np.loadtxt(file_name, dtype=np.int32)
        can_kg_np = np.unique(can_kg_np, axis=0)

        # remap ids in kg.
        can_kg_np = _remap_kg_id(can_kg_np)

        # get triplets with inverse direction like <entity, is-aspect-of, item>
        inv_kg_np = can_kg_np.copy()
        inv_kg_np[:, 0] = can_kg_np[:, 2]
        inv_kg_np[:, 2] = can_kg_np[:, 0]
        inv_kg_np[:, 1] = can_kg_np[:, 1] + max(can_kg_np[:, 1]) + 1

        # get full version of knowledge graph
        kg_np = np.concatenate((can_kg_np, inv_kg_np), axis=0)

        kg_dict, relation_dict = _construct_kg(kg_np)

        return kg_np, kg_dict, relation_dict

    def _statistic_kg_triples(self):
        def _id_range(kg_mat, idx):
            min_id = min(min(kg_mat[:, idx]), min(kg_mat[:, 2 - idx]))
            max_id = max(max(kg_mat[:, idx]), max(kg_mat[:, 2 - idx]))
            n_id = max_id - min_id + 1
            return (min_id, max_id), n_id

        self.entity_range, self.n_entities = _id_range(self.kg_data, idx=0)
        self.relation_range, self.n_relations = _id_range(self.kg_data, idx=1)
        self.n_kg_triples = len(self.kg_data)

        print("-" * 50)
        print(
            "-   entity_range: (%d, %d)" % (self.entity_range[0], self.entity_range[1])
        )
        print(
            "- relation_range: (%d, %d)"
            % (self.relation_range[0], self.relation_range[1])
        )
        print("-     n_entities: %d" % self.n_entities)
        print("-    n_relations: %d" % self.n_relations)
        print("-   n_kg_triples: %d" % self.n_kg_triples)
        print("-" * 50)


class CKGData(CFData, KGData):
    def __init__(self, args_config):
        CFData.__init__(self, args_config=args_config)
        KGData.__init__(
            self,
            args_config=args_config,
            entity_start_id=self.n_users,
            relation_start_id=2,
        )
        self.args_config = args_config

        self.ckg_graph = self._combine_cf_kg()

    def _combine_cf_kg(self):
        kg_mat = self.kg_data
        cf_mat = self.train_data

        # combine cf data and kg data:
        # ... ids of user entities in range of [0, #users)
        # ... ids of item entities in range of [#users, #users + #items)
        # ... ids of other entities in range of [#users + #items, #users + #entities)
        # ... ids of relations in range of [0, 2 + 2 * #kg relations), including two 'interact' and 'interacted_by'.
        ckg_graph = nx.MultiDiGraph()
        print("Begin to load interaction triples ...")
        for u_id, i_id in tqdm(cf_mat, ascii=True):
            ckg_graph.add_edges_from([(u_id, i_id)], r_id=0)
            ckg_graph.add_edges_from([(i_id, u_id)], r_id=1)

        print("\nBegin to load knowledge graph triples ...")
        for h_id, r_id, t_id in tqdm(kg_mat, ascii=True):
            ckg_graph.add_edges_from([(h_id, t_id)], r_id=r_id)
        return ckg_graph
