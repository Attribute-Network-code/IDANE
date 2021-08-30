import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from Utils.utils import *
from Model.SingleAE import SingleAE
import pickle

class Trainer(object):

    def __init__(self, model, config,graph):
        self.config = config
        self.model = model
        self.graph = graph
        self.net_input_dim = config['net_input_dim']
        self.att_input_dim = config['att_input_dim']
        self.adj_input_dim = config['adj_input_dim']
        self.adj_shape = config['adj_shape']
        self.net_shape = config['net_shape']
        self.att_shape = config['att_shape']
        self.drop_prob = config['drop_prob']
        self.beta = config['beta']
        self.gamma = config['gamma']
        self.alpha = config['alpha']
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.num_epochs = config['num_epochs']
        self.model_path = config['model_path']


        self.x = tf.placeholder(tf.float32, [None, self.net_input_dim])
        self.z = tf.placeholder(tf.float32, [None, self.att_input_dim])
        self.w = tf.placeholder(tf.float32, [None, self.adj_input_dim])
        self.rank = tf.placeholder(tf.float32, [None])
        # self.w = tf.placeholder(tf.float32, [None, None])

        self.neg_x = tf.placeholder(tf.float32, [None, self.net_input_dim])
        self.neg_z = tf.placeholder(tf.float32, [None, self.att_input_dim])
        self.neg_w = tf.placeholder(tf.float32, [None, self.adj_input_dim])
        self.neg_rank = tf.placeholder(tf.float32, [None])
        # self.neg_w = tf.placeholder(tf.float32, [None, None])

        self.optimizer, self.loss = self._build_training_graph()
        self.net_H, self.att_H, self.adj_H, self.H = self._build_eval_graph()

        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=gpu_config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
    # ///////////////////////////添加邻居表示/////////////////////////////////////////
    def build_neighbors_graph(self,X):
        # print("X = ",X)
        X_target = np.zeros(X.shape)
        nodes = self.graph.G.nodes();
        print("X_target.shape = ",X_target.shape)
        for node in nodes:
            neighbors = list(self.graph.G.neighbors(node))
            if len(neighbors) == 0:
                X_target[node] = X[node]
            else:
                temp = np.array(X[node])
                for item in neighbors:
                    temp = np.vstack((temp, X[item]))
                    pass
                temp = np.mean(temp, axis=0)
                X_target[node] = 1.0*temp+0*X[node]
            pass
        print("X_target.updata_shape = ", X_target.shape)
        return X_target;
        pass

    # ///////////////////////////添加邻居表示/////////////////////////////////////////

    # ///////////////////////////添加节点排名表示/////////////////////////////////////////
    def node_rank(self):
        nodes = self.graph.G.nodes();
        nodes_degree = {}
        nodeRank = []
        for node in nodes:
            neighbors = list(self.graph.G.neighbors(node))
            nodes_degree[node] = len(neighbors)#记录下每个节点的度
        for node in nodes:
            neighbors = list(self.graph.G.neighbors(node))
            # sum =nodes_degree[node];
            sum =0;
            for item in neighbors:
                sum += 1.0/nodes_degree[item];#方案1；
                # sum += nodes_degree[item];#方案2
            # print("sum = ",sum);
            # print("nodes_degree[node] = ",nodes_degree[node])
            if(sum ==0):
                nodeRank.append(1)
            else:
                sum = sum / len(neighbors) #方案1；
                # sum = len(neighbors) / sum;  # 方案2；
                nodeRank.append(sum)
            pass
        return nodeRank,nodes_degree
        pass
    # ///////////////////////////添加节点排名表示/////////////////////////////////////////

    def _build_training_graph(self):
        net_H, net_recon = self.model.forward_net(self.x, drop_prob=self.drop_prob, reuse=False)
        neg_net_H, neg_net_recon = self.model.forward_net(self.neg_x, drop_prob=self.drop_prob, reuse=True)

        adj_H, adj_recon = self.model.forward_adj(self.w, drop_prob=self.drop_prob, reuse=False)
        neg_adj_H, neg_adj_recon = self.model.forward_adj(self.neg_w, drop_prob=self.drop_prob, reuse=True)

        att_H, att_recon = self.model.forward_att(self.z, drop_prob=self.drop_prob, reuse=False)
        neg_att_H, neg_att_recon = self.model.forward_att(self.neg_z, drop_prob=self.drop_prob, reuse=True)


        #================high-order proximity & semantic proximity=============
        print("net_recon.shape = ", net_recon.shape)
        print("net_recon = ", net_recon)
        print("self.x.shape = ", self.x.shape)
        print("self.x - net_recon = ", (self.x - net_recon))

        # recon_loss_1 = tf.reduce_mean(tf.reduce_sum(tf.square(self.x - net_recon), 1))
        # recon_loss_2 = tf.reduce_mean(tf.reduce_sum(tf.square(self.neg_x - neg_net_recon), 1))
        # recon_loss_3 = tf.reduce_mean(tf.reduce_sum(tf.square(self.z - att_recon), 1))
        # recon_loss_4 = tf.reduce_mean(tf.reduce_sum(tf.square(self.neg_z - neg_att_recon), 1))
        # recon_loss_5 = tf.reduce_mean(tf.reduce_sum(tf.square(self.w - adj_recon), 1))
        # recon_loss_6 = tf.reduce_mean(tf.reduce_sum(tf.square(self.neg_w - neg_adj_recon), 1))
        #
        recon_loss_1 = tf.reduce_mean(tf.reduce_sum(tf.square(((self.x - net_recon) * self.rank[:,None])), 1))
        recon_loss_2 = tf.reduce_mean(tf.reduce_sum(tf.square(((self.neg_x - neg_net_recon) * self.neg_rank[:,None])), 1))
        recon_loss_3 = tf.reduce_mean(tf.reduce_sum(tf.square(((self.z - att_recon) * self.rank[:,None])), 1))
        recon_loss_4 = tf.reduce_mean(tf.reduce_sum(tf.square(((self.neg_z - neg_att_recon) * self.neg_rank[:,None])), 1))
        recon_loss_5 = tf.reduce_mean(tf.reduce_sum(tf.square(((self.w - adj_recon) * self.rank[:,None])), 1))
        recon_loss_6 = tf.reduce_mean(tf.reduce_sum(tf.square(((self.neg_w - neg_adj_recon) * self.neg_rank[:,None])), 1))
        recon_loss = recon_loss_1 + recon_loss_2 + recon_loss_3 + recon_loss_4 + recon_loss_5 + recon_loss_6


        #===============cross modality proximity==================
        pre_logit_pos = tf.reduce_sum(tf.multiply(net_H, att_H), 1)
        pre_logit_neg_1 = tf.reduce_sum(tf.multiply(neg_net_H, att_H), 1)
        pre_logit_neg_2 = tf.reduce_sum(tf.multiply(net_H, neg_att_H), 1)
        pre_logit_pos2 = tf.reduce_sum(tf.multiply(adj_H, att_H), 1)
        pre_logit_neg_21 = tf.reduce_sum(tf.multiply(neg_adj_H, att_H), 1)
        pre_logit_neg_22 = tf.reduce_sum(tf.multiply(adj_H, neg_att_H), 1)

        pos_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(pre_logit_pos), logits=pre_logit_pos)
        neg_loss_1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(pre_logit_neg_1), logits=pre_logit_neg_1)
        neg_loss_2 = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(pre_logit_neg_2), logits=pre_logit_neg_2)
        pos_loss2 = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(pre_logit_pos2), logits=pre_logit_pos2)
        neg_loss_21 = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(pre_logit_neg_21), logits=pre_logit_neg_21)
        neg_loss_22 = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(pre_logit_neg_22), logits=pre_logit_neg_22)

        cross_modal_loss = tf.reduce_mean(pos_loss + neg_loss_1 + neg_loss_2 + pos_loss2 + neg_loss_21 + neg_loss_22)





        #==========================================================
        loss = recon_loss * self.beta + cross_modal_loss * self.alpha
        # loss = recon_loss * self.beta + first_order_loss * self.gamma + cross_modal_loss * self.alpha


        vars_net = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net_encoder')
        vars_att = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'att_encoder')
        vars_adj = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'adj_encoder')
        print(vars_net)


        opt = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, var_list=vars_net+vars_att+vars_adj)

        return opt, loss

    def _build_eval_graph(self):
        net_H, _ = self.model.forward_net(self.x, drop_prob=0.0, reuse=True)
        att_H, _ = self.model.forward_att(self.z, drop_prob=0.0, reuse=True)
        adj_H, _ = self.model.forward_adj(self.w, drop_prob=0.0, reuse=True)
        H = tf.concat([tf.nn.l2_normalize(net_H, dim=1), tf.nn.l2_normalize(att_H, dim=1), tf.nn.l2_normalize(adj_H, dim=1)], axis=1)

        return net_H, att_H, adj_H, H

    def sample_by_idx_by_neighbor(self,idx):
        nodeRank, _ = self.node_rank();
        # print("nodeRank = ", nodeRank)
        # print("nodeRank.shape() = ", len(nodeRank))
        mini_batch = Dotdict()
        mini_batch.X = self.graph.X[idx]
        mini_batch.Z = self.neighbor_Z[idx]
        mini_batch.W = self.neighbor_W[idx]
        # print("idx = ", idx)
        mini_batch.rank =[]
        for item in idx:
            mini_batch.rank.append(nodeRank[item]);


        return mini_batch

    def train(self, graph):
        print('///////////////////////////////////////////////////')
        # print(self.build_neighbors_graph(graph.X))
        # print(self.build_neighbors_graph(graph.W))
        # print(self.build_neighbors_graph(graph.Z))
        nodeRank,_ =self.node_rank();
        self.neighbor_X = self.build_neighbors_graph(graph.X);
        self.neighbor_Z = self.build_neighbors_graph(graph.Z);
        self.neighbor_W= self.build_neighbors_graph(graph.W);

        print('///////////////////////////////////////////////////')
        for epoch in range(self.num_epochs):

            idx1, idx2 = self.generate_samples(graph)

            index = 0
            cost = 0.0
            cnt = 0
            while True:
                if index > graph.num_nodes:
                    break
                if index + self.batch_size < graph.num_nodes:
                    mini_batch1 = self.sample_by_idx_by_neighbor(idx1[index:index + self.batch_size])
                    mini_batch2 = self.sample_by_idx_by_neighbor(idx2[index:index + self.batch_size])
                    # mini_batch1 = graph.sample_by_idx(idx1[index:index + self.batch_size])
                    # mini_batch2 = graph.sample_by_idx(idx2[index:index + self.batch_size])
                else:
                    mini_batch1 = self.sample_by_idx_by_neighbor(idx1[index:])
                    mini_batch2 = self.sample_by_idx_by_neighbor(idx2[index:])
                    # mini_batch1 = graph.sample_by_idx(idx1[index:])
                    # mini_batch2 = graph.sample_by_idx(idx2[index:])
                index += self.batch_size
                # print(mini_batch1.X.shape)

                # print("mini_batch1.x.shape = ",mini_batch1.x)
                loss, _ = self.sess.run([self.loss, self.optimizer],
                                        feed_dict={self.x: mini_batch1.X,
                                                   self.z: mini_batch1.Z,
                                                   self.rank: mini_batch1.rank,
                                                   self.neg_x: mini_batch2.X,
                                                   self.neg_z: mini_batch2.Z,
                                                   self.neg_rank:mini_batch2.rank,
                                                   self.w: mini_batch1.W,
                                                   self.neg_w: mini_batch2.W})

                cost += loss
                cnt += 1

                if graph.is_epoch_end:
                    break
            cost /= cnt

            if epoch % 50 == 0:

                train_emb = None
                train_label = None
                while True:
                    mini_batch = graph.sample(self.batch_size, do_shuffle=False, with_label=True)

                    emb = self.sess.run(self.H,
                                        feed_dict={self.x: mini_batch.X,
                                                   self.z: mini_batch.Z,
                                                   self.w: mini_batch.W})
                    if train_emb is None:
                        train_emb = emb
                        train_label = mini_batch.Y
                    else:
                        train_emb = np.vstack((train_emb, emb))
                        train_label = np.vstack((train_label, mini_batch.Y))

                    if graph.is_epoch_end:
                        break
                micro_f1, macro_f1 = check_multi_label_classification(train_emb, train_label, 0.5)
                print('Epoch-{}, loss: {:.4f}, Micro_f1 {:.4f}, Macro_fa {:.4f}'.format(epoch, cost, micro_f1, macro_f1))

        self.save_model()


    def infer(self, graph):
        self.sess.run(tf.global_variables_initializer())
        self.restore_model()
        print("Model restored from file: %s" % self.model_path)

        train_emb = None
        train_label = None
        while True:
            mini_batch = graph.sample(self.batch_size, do_shuffle=False, with_label=True)
            emb = self.sess.run(self.H, feed_dict={self.x: mini_batch.X,
                                                   self.z: mini_batch.Z,
                                                   self.w: mini_batch.W})

            if train_emb is None:
                train_emb = emb
                train_label = mini_batch.Y
            else:
                train_emb = np.vstack((train_emb, emb))
                train_label = np.vstack((train_label, mini_batch.Y))

            if graph.is_epoch_end:
                break


        # test_ratio = np.arange(0.5, 1.0, 0.2)
        test_ratio = np.arange(0.5, 1.0, 0.1)
        dane = []
        for tr in test_ratio[-1::-1]:
            print('============train ration-{}=========='.format(1 - tr))
            micro, macro = multi_label_classification(train_emb, train_label, tr)
            dane.append('{:.4f}'.format(micro) + ' & ' + '{:.4f}'.format(macro))
        print(' & '.join(dane))



    def generate_samples(self, graph):
        X = []
        Z = []
        W = []

        order = np.arange(graph.num_nodes)
        np.random.shuffle(order)

        index = 0
        while True:
            if index > graph.num_nodes:
                break
            if index + self.batch_size < graph.num_nodes:
                mini_batch = graph.sample_by_idx(order[index:index + self.batch_size])
            else:
                mini_batch = graph.sample_by_idx(order[index:])
            index += self.batch_size

            net_H, att_H, adj_H = self.sess.run([self.net_H, self.att_H,self.adj_H],
                                         feed_dict={self.x: mini_batch.X,
                                                    self.z: mini_batch.Z,
                                                    self.w: mini_batch.W})
            X.extend(net_H)
            Z.extend(att_H)
            W.extend(adj_H)

        X = np.array(X)
        Z = np.array(Z)
        W = np.array(W)

        X = preprocessing.normalize(X, norm='l2')
        Z = preprocessing.normalize(Z, norm='l2')
        W = preprocessing.normalize(W, norm='l2')

        sim = np.dot(X, Z.T) + np.dot(W, Z.T)

        neg_idx = np.argmin(sim, axis=1)


        return order, neg_idx


    def save_model(self):
        self.saver.save(self.sess, self.model_path)

    def restore_model(self):
        self.saver.restore(self.sess, self.model_path)
