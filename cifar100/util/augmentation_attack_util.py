from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K


def get_max_accuracy(y_true, probs, thresholds=None):
  """Return the max accuracy possible given the correct labels and guesses. Will try all thresholds unless passed.

  Args:
    y_true: True label of `in' or `out' (member or non-member, 1/0)
    probs: The scalar to threshold
    thresholds: In a blackbox setup with a shadow/source model, the threshold obtained by the source model can be passed
      here for attackin the target model. This threshold will then be used.

  Returns: max accuracy possible, accuracy at the threshold passed (if one was passed), the max precision possible,
   and the precision at the threshold passed.

  """
  if thresholds is None:
    fpr, tpr, thresholds = roc_curve(y_true, probs)

  accuracy_scores = []
  precision_scores = []
  for thresh in thresholds:
    accuracy_scores.append(accuracy_score(y_true,
                                          [1 if m > thresh else 0 for m in probs]))
    precision_scores.append(precision_score(y_true, [1 if m > thresh else 0 for m in probs]))

  accuracies = np.array(accuracy_scores)
  precisions = np.array(precision_scores)
  max_accuracy = accuracies.max()
  max_precision = precisions.max()
  max_accuracy_threshold = thresholds[accuracies.argmax()]
  max_precision_threshold = thresholds[precisions.argmax()]
  return max_accuracy, max_accuracy_threshold, max_precision, max_precision_threshold


def get_threshold(source_m, source_stats, target_m, target_stats):
  """ Train a threshold attack model and get teh accuracy on source and target models.

  Args:
    source_m: membership labels for source dataset (1 for member, 0 for non-member)
    source_stats: scalar values to threshold (attack features) for source dataset
                  Here it means the robust accuracy proxy on the noisy neighbours

                  
    target_m: membership labels for target dataset (1 for member, 0 for non-member)
    target_stats: scalar values to threshold (attack features) for target dataset

  Returns: best acc from source thresh, precision @ same threshold, threshold for best acc,
    precision at the best threshold for precision. all tuned on source model.

  """
  # find best threshold on source data
  acc_source, t, prec_source, tprec = get_max_accuracy(source_m, source_stats)

  # find best accuracy on test data (just to check how much we overfit)
  acc_test, _, prec_test, _ = get_max_accuracy(target_m, target_stats)

  # get the test accuracy at the threshold selected on the source data
  acc_test_t, _, _, _ = get_max_accuracy(target_m, target_stats, thresholds=[t])
  _, _, prec_test_t, _ = get_max_accuracy(target_m, target_stats, thresholds=[tprec])
  print("acc src: {}, acc test (best thresh): {}, acc test (src thresh): {}, thresh: {}".format(acc_source, acc_test,
                                                                                                acc_test_t, t))
  print(
    "prec src: {}, prec test (best thresh): {}, prec test (src thresh): {}, thresh: {}".format(prec_source, prec_test,
                                                                                               prec_test_t, tprec))

  return acc_test_t, prec_test_t, t, tprec


def train_model(train_set, **kwargs):
  models = [AttackModel(kwargs.get('type_')) for _ in range(kwargs.get('n_classes', 10))]
  for model in models:
    model.compile(tf.keras.optimizers.Adam(
      learning_rate=kwargs.get('learning_rate', 1e-3)),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(),
      ])
  epochs = []
  losses = []
  for i in range(len(models)):
    train_lbl_sel = train_set[1].flatten() == i
    if np.sum(train_lbl_sel.astype(np.int)) <= 0.:
      raise ValueError(f'No data of label: {i}')
    # val_lbl_sel = val_set[1] == i
    # mem_sel = np.squeeze(train_set[2][train_lbl_sel]) == 1
    # print(f"TRAIN m=0 x: {np.mean(train_set[0][train_lbl_sel][~mem_sel], axis=0)}")
    # print(f"TRAIN m=1 x: {np.mean(train_set[0][train_lbl_sel][mem_sel], axis=0)}")
    # mem_sel = np.squeeze(val_set[2][val_lbl_sel]) == 1
    # print(f"VAL m=0 x: {np.mean(val_set[0][val_lbl_sel][~mem_sel], axis=0)}")
    # print(f"VAL m=1 x: {np.mean(val_set[0][val_lbl_sel][mem_sel], axis=0)}")
    history = models[i].fit(train_set[0][train_lbl_sel],
                            train_set[2][train_lbl_sel],
                            batch_size=kwargs.get('batch_size', 1),
                            # callbacks=[tf.keras.callbacks.EarlyStopping(
                            #   monitor='val_loss', mode='min', patience=10)],
                            # validation_data=(val_set[0][val_lbl_sel],
                            #                  val_set[2][val_lbl_sel]),
                            epochs=kwargs.get('epochs', 1),
                            shuffle=True, verbose=0,
                            )
    hist = history.history
    keys = hist.keys()
    hist = {key: hist[key][-1] for key in keys if key == 'loss' or key == 'sparse_categorical_accuracy'}
    epochs.append(len(history.history['loss']))
    losses.append(hist['loss'])
    # print(f"label: '{i}' has {hist}")
  return models, np.mean(losses)


def calc_confuse(preds, labels):
  labels = labels.astype(np.bool).squeeze()
  preds = np.argmax(preds, axis=1).astype(np.bool)
  tp = np.logical_and(np.equal(labels, True), np.equal(preds, True)).astype(
    np.int).sum()
  fp = np.logical_and(np.equal(labels, False), np.equal(preds, True)).astype(
    np.int).sum()
  tn = np.logical_and(np.equal(labels, False), np.equal(preds, False)).astype(
    np.int).sum()
  fn = np.logical_and(np.equal(labels, True), np.equal(preds, False)).astype(
    np.int).sum()
  return tp, fp, tn, fn


def test_model(models, test_set, type_, toprint=True):
  tps, fps, tns, fns, lens = [], [], [], [], []
  aucs = []
  preds = []
  accs = []
  recs = []
  precs = [] 

  re_ordered_membership_labels = []
  score_for_ground_truth_cls = []

  for i in range(len(models)):
    lbl_sel = test_set[1].flatten() == i # each label corresponds to one model
                                          # each model is doing inference on the corresponding inputs only (not all inputs)
    features = test_set[0][lbl_sel]
    membership_labels = test_set[2][lbl_sel]
    lens.append(len(membership_labels))
    loss, acc = models[i].evaluate(features,
                                   membership_labels,
                                   batch_size=1000,
                                   verbose=0)
    pred = models[i].predict(features)
    tp, fp, tn, fn = calc_confuse(pred, membership_labels)
 
    # calcacc = (tp + tn) / (tp + tn + fp + fn)
    accs.append(acc)
    tps.append(tp)
    fps.append(fp)
    tns.append(tn)
    fns.append(fn)
    preds.extend(pred) 
 


    # pred is two dim [is_non_member, is_member]
    # we record the pred score of the input being member
    # if this score is > 0.5, then predict member, otherwise non-member
    for j in range( len(pred) ):
      score_for_ground_truth_cls.append( pred[j][1] )
 
    re_ordered_membership_labels.extend(membership_labels) # this will be used for computing the auc
   
  #print( np.asarray(score_for_ground_truth_cls).shape, np.asarray(re_ordered_membership_labels).shape )
  #tmp_accuracy =  np.where( ( np.asarray(score_for_ground_truth_cls)>0.5) == np.asarray(re_ordered_membership_labels) )[0] 
  #print( 'acy computed using 0.5 threshold ==> ', len(tmp_accuracy) / float( len(score_for_ground_truth_cls) ) )


  accuracy = (np.sum(tps) + np.sum(tns)) / (
      np.sum(tps) + np.sum(tns) + np.sum(fps) + np.sum(fns))
  r = np.sum(tps) / (np.sum(tps) + np.sum(fns))
  p = np.sum(tps) / (np.sum(tps) + np.sum(fps))
  f1 = (2 * (r * p)) / (r + p)
  auc = 0
  if toprint:
    print("{}|: accuracy: {}. F1: {}, prec: {}, recall {}'".format(type_, accuracy, f1, p, r))

  return accuracy, auc, f1, accs, aucs, precs, recs, p, r, score_for_ground_truth_cls, re_ordered_membership_labels


def assign_best(best, testbest, metric, val, other_vals, old_models, new_models, testval, testothers):
  if val > best[metric]:
    best[metric] = val
    testbest[metric] = testval
    for (key, v) in other_vals:
      best[key] = v
    for (key, v) in testothers:
      testbest[key] = v
    return new_models
  return old_models


class AttackModel(tf.keras.Model):
  def __init__(self, aug_type='n'):
    """ Sample Attack Model.

    :param aug_type:
    """
    super().__init__()
    if aug_type == 'n':
      self.x1 = tf.keras.layers.Dense(64, activation=tf.keras.layers.ReLU(
        negative_slope=1e-2), kernel_initializer='glorot_normal')
      self.x_out = tf.keras.layers.Dense(2, kernel_initializer='glorot_normal')

    elif aug_type == 'r' or aug_type == 'd':
      self.x1 = tf.keras.layers.Dense(10, activation=tf.keras.layers.ReLU(
        negative_slope=1e-2), kernel_initializer='glorot_normal')
      self.x2 = tf.keras.layers.Dense(10, activation=tf.keras.layers.ReLU(
        negative_slope=1e-2), kernel_initializer='glorot_normal')
      self.x_out = tf.keras.layers.Dense(2, kernel_initializer='glorot_normal')
    else:
      raise ValueError(f"aug_type={aug_type} is not valid.")

    self.x_activation = tf.keras.layers.Softmax()

  def call(self, inputs, training=False):
    x = inputs
    for layer in self.layers:
      x = layer(x)
    return x


def train_best_attack_model(train_set, test_set, type_, n_classes=10):
  val_best = {'acc': -1, 'f1': -1, 'prec': -1, 'recl' : -1}
  test_best = {'acc': -1, 'f1': -1, 'prec': -1, 'recl' : -1}

  l = np.array(
    [1 * 10 ** i for i in range(-3, -1)] + [5 * 10 ** i for i in range(-3, -1)])
  best_models = None

  preds_per_lr = []
  labels_per_lr = []
  for i in range(len(l)):
    print("attack model with lr of %f"%l[i])
    print()
    K.clear_session()
    learning_rate = l[i]
    models, epochs = train_model(train_set, learning_rate=learning_rate, type_=type_, n_classes=n_classes)
    acc, _, f1, _, _, _, _, p, r, _, _ = test_model(models, train_set, 'source')
    testacc, _, testf1, _, _, _, _, testp, testr, preds, re_ordered_membership_labels  = test_model(models, test_set, 'target')

    best_models = assign_best(val_best, test_best, 'acc', acc, [('f1', f1), ('prec', p), ('recl', r)], best_models, models, testacc,
                              [('f1', testf1), ('prec', testp), ('recl', testr)])
    preds_per_lr.append(preds)
    labels_per_lr.append(re_ordered_membership_labels)


  return preds_per_lr, labels_per_lr, test_best

 



