# -*- coding: utf-8 -*-
"""
Created on Mon Dec 04 11:05:31 2017

@author: lming
"""
# LEARN WITHOUT DREAMING?
import gc

from keras.utils import to_categorical

from calagent import CALAgent
from tagger import CRFTagger
import utilities
import tensorflow as tf
import numpy as np
import time
import os

args = utilities.get_args()
logger = utilities.init_logger()

max_len = args.max_seq_length
VOCABULARY= args.vocab_size
EPISODES=args.episodes
BUDGET=args.annotation_budget
k=args.k

rootdir=args.root_dir
train_file=args.train_file
dev_file=args.dev_file
test_file=args.test_file
emb_file= args.word_vec_file
DATASET_NAME=args.dataset_name
policy_output="{}/{}_policy.ckpt".format(args.output, DATASET_NAME)
tagger_output="{}/{}_tagger.h5".format(args.output, DATASET_NAME)
tagger_temp = "{}/{}_tagger_temp.h5".format(args.output, DATASET_NAME)

logger.info("Train AL policy on dataset {} in {} episode with budget {}".format(DATASET_NAME, EPISODES, BUDGET))
logger.info(" * Vocabulary size: {}".format(VOCABULARY))
logger.info(" * INPUT train file: {}".format(train_file))
logger.info(" * INPUT dev file: {}".format(dev_file))
logger.info(" * INPUT test file: {}".format(test_file))
logger.info(" * POLICY OUTPUT path: {}".format(policy_output))
logger.info(" * TAGGER OUTPUT file: {}".format(tagger_output))

resultname=args.output
logger.info("Processing data")
if args.ber_task:
    logger.info("Processing data for BER task")
    train_x, train_y, train_lens = utilities.load_data2labels_BER(train_file)
    test_x, test_y, test_lens = utilities.load_data2labels_BER(test_file)
    dev_x, dev_y, dev_lens = utilities.load_data2labels_BER(dev_file)
    label2str = utilities.BER_label2str
    num_classes = 6
elif args.ibo_scheme:
    train_x, train_y, train_lens = utilities.load_data2labels_IBO(train_file)
    test_x, test_y, test_lens = utilities.load_data2labels_IBO(test_file)
    dev_x, dev_y, dev_lens = utilities.load_data2labels_IBO(dev_file)
    num_classes = 9
    label2str = utilities.IBO_label2str
else:
    train_x, train_y, train_lens = utilities.load_data2labels_IO(train_file)
    test_x, test_y, test_lens = utilities.load_data2labels_IO(test_file)
    dev_x, dev_y, dev_lens = utilities.load_data2labels_IO(dev_file)
    num_classes = 5
    label2str = utilities.IO_label2str

# train, test, dev
train_sents = utilities.data2sents(train_x, train_y)
test_sents = utilities.data2sents(test_x, test_y)
dev_sents = utilities.data2sents(dev_x, dev_y)

# build vocabulary
logger.info("Max document length: {}".format(max_len))
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        max_document_length=max_len, min_frequency=1)
# vocab = vocab_processor.vocabulary_ # start from {"<UNK>":0}
train_idx = list(vocab_processor.fit_transform(train_x))
dev_idx = list(vocab_processor.fit_transform(dev_x))
vocab = vocab_processor.vocabulary_
vocab.freeze()
test_idx = list(vocab_processor.fit_transform(test_x))

allsents=train_sents + dev_sents + test_sents
allidx=train_idx + dev_idx + test_idx

# build embeddings
vocab = vocab_processor.vocabulary_
vocab_size = VOCABULARY
w2v = utilities.load_crosslingual_embeddings(emb_file, vocab, vocab_size)
embedding_size = np.shape(w2v)[1]
payments = [0.02,0.08,0.14,0.2,0.26]
c = len(payments)
agent=CALAgent(max_len, embedding_size, vocab_size, np.asarray(w2v), num_classes, policy_output, payments)
start_time = time.time()
logger.info('Fixed beta = 0.5')

states=[]
reward_states=[]
actions=[]
costs=[]
#Training the policy
for tau in range(0, EPISODES):
    #Shuffle train_sents, split into train_la and train_pool
    indices = np.arange(len(allsents))
    np.random.shuffle(indices)
    train_la=[]
    train_val=[]
    train_pool=[]
    train_la_idx=[]
    train_pool_idx=[]
    for i in range(0,len(allsents)):
        if(i<10):
            train_la.append(allsents[indices[i]])
            train_la_idx.append(allidx[indices[i]])
        if(i>10 and i<(len(dev_sents)+10)):
            train_val.append(allsents[indices[i]])
        else:
            train_pool.append(allsents[indices[i]])
            train_pool_idx.append(allidx[indices[i]])

    #Initialise the model
    if os.path.exists(tagger_output):
        os.remove(tagger_output)
    model = CRFTagger(tagger_output, num_classes=num_classes)
    if args.initial_training_size > 0:
        model.train(train_la)
    # sample a cost-accuracy trade-off
    upp=0.4
    low=0.8
    para = (np.random.uniform()-0.5) * (upp-low)/2 + (upp+low)/2
    accuracy = [utilities.para_curve(cost, 'linear',p=para) for cost in payments]
    # flip labels according to accuracy and save it 
    flipped_labels = dict()
    for cost, acc in zip(payments, accuracy):
        flipped_labels[cost] = utilities.flip_labels_diag(train_pool, acc, num_classes)
    # start the episode
    t = 0
    while t < BUDGET:
    #for t in range(0, BUDGET):
        #if (t % 10) == 0:
        logger.info(' * Episode: {} Budget so far: {}'.format(str(tau+1), str(t)))
        #Random get k sample from train_pool and train_pool_idx
        random_pool, random_pool_idx, queryindices= utilities.randomKSamples(train_pool,train_pool_idx,k)
        row=0
        f1=-1
        bestindex=0
        newseq=[]
        newidx=[]
        coin=np.random.rand(1) # beta=0.5 fixed
        #coin=max(0.5,1-0.01*tau)  #beta=max(0.5,1-0.01*tau) linear decay
        # coin=0.9**tau             #beta= 0.9**tau           exponential decay
        #coin=5/(5+np.exp(tau/5))  #beta=5/(5+exp(tau/5))    inverse sigmoid decay

        #get the state and action
        state=utilities.getAllState(train_la_idx, random_pool,random_pool_idx, model, w2v, max_len, num_classes,payments)
 
        # find thd best instance in f1 score in the random pool 
        for datapoint in zip(random_pool,random_pool_idx):
            for pay in payments:
                seq=datapoint[0]
                idx=datapoint[1]
                #seq = flipped_labels[pay][idx]
                seq = utilities.flip_seq(seq, utilities.para_curve(pay, 'linear',p=para), num_classes)
                #train_la_temp=[]
                train_la_temp=list(train_la)
                train_la_temp.append(seq)

                if os.path.exists(tagger_temp):
                    os.remove(tagger_temp)
                model_temp=CRFTagger(tagger_temp, num_classes=num_classes)
                model_temp.train(train_la_temp)
                f1_temp=model_temp.test(train_val, label2str)
                # reward network
                cost_input = pay
                f1_temp=agent.predict_reward(f1_temp, state[0][row:(row+1)], state[1][row:(row+1)], state[2][row:(row+1)], state[3][row:(row+1)], state[4][row:(row+1)], state[5][row:(row+1)], np.asarray([[cost_input]]))
                if(f1_temp>f1):
                    bestindex=row
                    bestcost=pay
                    f1=f1_temp
                    newseq=seq
                    newidx=idx
                del model_temp
                del train_la_temp
                gc.collect()
            row += 1
        #get the state and action
        #state=utilities.getAllState(train_la_idx, random_pool,random_pool_idx, model, w2v, max_len, num_classes)
        
        #action=bestindex
        if(coin>0.5):
            action, cost = bestindex, bestcost
            #tempstates= np.ndarray((1,K,len(state[0])), buffer=np.array(state))
        else:
            # sent_content= np.expand_dims(state[0], axis=0)
            # marginal_prob = np.expand_dims(state[1], axis=0)
            # confidence_score = np.expand_dims(state[2], axis=0)
            # labeled_data = np.expand_dims(state[3], axis=0)
            action, cost = agent.predict_action(k, len(payments), state[0], state[1], state[2], state[3], state[4], state[5], state[6])
            cost = payments[cost]
        reward_states.append([state[i][action] for i in range(6)])
        states.append(state)
        c_action = action + k * np.where(cost==np.array(payments))[0][0]
        actions.append(c_action)
        costs.append(cost)
        t += cost
        #update the model

        # select the best index from rand pool
        #delete the selected data point from the pool
        theindex=queryindices[bestindex]

        # append
        train_la.append(train_pool[theindex])
        train_la_idx.append(train_pool_idx[theindex])
        # model retrain
        model.train(train_la)

        del train_pool[theindex]
        del train_pool_idx[theindex]  
    # end of the episode
    f1_final=model.test(train_val, label2str)
    logger.info('F1 score: {}'.format(str(f1_final)))
    f1_final=[f1_final]*len(states)
    agent.train_reward(reward_states, costs, f1_final)

    #train the policy
    # states=np.array(states)
    cur_actions=to_categorical(np.asarray(actions), num_classes=args.k*c)
    #state-action pairs in the episode
    # actions dimension k*c
    agent.train_policy(k, c, states, cur_actions)

#save the policy
logger.info("Train:--- %s seconds ---" % (time.time() - start_time))

