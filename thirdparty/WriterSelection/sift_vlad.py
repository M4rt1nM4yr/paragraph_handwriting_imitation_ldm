import pickle
import os
import shlex
import argparse
from tqdm import tqdm
import gzip
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize, StandardScaler
import numpy as np
import numpy.ma as ma

import cv2
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC
import h5py
import uuid

#from reranking import sgr
import torch

from src.utils.utils import *
from src.data.utils.constants import *
import torch

class WriterSelect:
    def __init__(self,train_config,tmp_folder, pca_comps=-1,ipca_comps=-1,powernorm=True,
                 patch_size=None,n_clusters=100,overwrite=False,gmp=None,esvm=True,standardize=None,C=1000,n_mvlad=1,
                 gamma = 1000,iesvm=False,application_reference_dl=r"Dataloaders/768x768"):

        super().__init__()
        self.iesvm = iesvm
        self.pca_comps = pca_comps
        self.ipca_comps = ipca_comps
        self.powernorm = powernorm
        self.patch_size = patch_size
        self.n_clusters = n_clusters
        self.gmp = gmp
        self.esvm = esvm
        self.standardize = standardize
        self.C = C
        self.n_mvlad = n_mvlad
        self.tmp_folder = tmp_folder
        self.gamma = gamma


        np.random.seed(0)  # fix random seed
        # load files of training
        # TODO rewrite this so I can use it with my dataloaders instead ....
        #  files_train, labels_train = getFiles(args.in_train, args.suffix_train,
        #                                       args.labels_train)

        files_train, labels_train = get_data_from_dataloader(train_config,application=application_reference_dl)
        assert (len(files_train) == len(labels_train))
        print('# train:', len(files_train))

        pca_v = None
        pca = None
        self.ipcas = []
        self.all_enc_train = []
        self.all_mus = []
        for i in range(n_mvlad):
            # a) dictionary
            print('> compute dictionary')
            descriptors = runF('rand_descs_{}.pkl.gz'.format(i), overwrite, tmp_folder,
                               loadRandomDescriptors, files_train,300000,# 150000,
                               patch_size)
            print('> loaded/computed {} descriptors:'.format(len(descriptors)))

            if ipca_comps >= 0:
                ipca_comps = pca_comps if pca_comps > 0 \
                    else descriptors.shape[1]
                print(ipca_comps)
                pca = PCA(ipca_comps, whiten=True)
                descriptors = pca.fit_transform(descriptors)
                descriptors = powernormalize(descriptors)
                self.ipcas.append(pca)

            mus = runF('dict_{}.pkl.gz'.format(i), overwrite, tmp_folder,
                       dictionary, descriptors, n_clusters)
            self.all_mus.append(mus)

            # vlad w. PCA
            #    a = assignments(desc, mus)
            #    T, D = desc.shape
            #    f_enc = np.zeros((T,D * args.n_clusters), dtype=np.float32)
            #    for k in tqdm(range(T)):
            #        ind = np.nonzero(a[k,:])[0][0]
            #        f_enc[k, ind:ind+D] = desc[k] - mus[ind]
            #    print('run PCA')
            #    pca_v = PCA(2048)
            #    pca_v.fit(f_enc)

            # c) VLAD encoding training
            print('> compute VLAD for train')
            enc_train = runF('encs_train_vlad_{}.pkl.gz'.format(i), overwrite,
                             tmp_folder,
                             vlad, files_train, mus, powernorm, gmp, 1000,
                             patch_size, pca)

            # TODO: think of how to do e-svm and rerank for train
            # does this just work like that?
            # NO -> because it changes the enc_trains already!
            #        if args.iesvm:
            #            print('> intermediate esvm computation')
            #            # TODO: for a real setup you actually want to cross-validate C
            #    #        all_enc_test = esvm(all_enc_test, all_enc_train, args.C)
            #            enc_train = runF('encs_train_esvm_{}.pkl.gz'.format(i), args.overwrite,
            #                            args.tmp_folder,
            #                            esvm, enc_train, enc_train, args.C, True)
            #
            self.all_enc_train.append(enc_train)


    def prepare_data(self,files_test,label):
        labels = []

        if len(files_test[0].shape) ==3:
            files_tmp = []
            for x in files_test:
                files_tmp.append(np.squeeze(x))
                labels.append(label)
            labels = np.asarray(labels)

        return files_tmp,labels


    def tests(self,generated_samples, style_sample,labels_test,prepare_data=False):


        pca_v = None
        indices = None

        files_test = generated_samples #+ [style_sample]

        style_sample = [style_sample.squeeze()]
        style_labels = np.array([labels_test])

        if prepare_data:
            files_test, labels_test = self.prepare_data(files_test,labels_test)

        #    # b) VLAD encoding for test
        all_enc_test = []
        style_enc = []
        for i in range(self.n_mvlad):
            print('run ', i)
            enc_test = runF('encs_test_vlad_{}.pkl.gz'.format(i), True,
                            self.tmp_folder,
                            vlad, files_test, self.all_mus[i], self.powernorm, self.gmp, self.gamma,
                            self.patch_size, self.ipcas[i] if self.ipca_comps >= 0 else None, pca_v=pca_v)

            enc_style = runF('encs_style_vlad_{}.pkl.gz'.format(i), True,
                            self.tmp_folder,
                            vlad, style_sample, self.all_mus[i], self.powernorm, self.gmp, self.gamma,
                            self.patch_size, self.ipcas[i] if self.ipca_comps >= 0 else None, pca_v=pca_v)
        #    evaluate(enc_test, labels_test)


            # save here the original ones! not the e-svm transformed
            all_enc_test.append(enc_test)
            style_enc.append(enc_style)

        # TODO
        all_enc_train = np.concatenate(self.all_enc_train, axis=1)
        all_enc_test = np.concatenate(all_enc_test, axis=1)
        style_enc = np.concatenate(style_enc,axis=1)

        all_enc_test = np.concatenate((all_enc_test,style_enc))
        #
        if self.pca_comps >= 0:
            print('fit PCA')
            if self.pca_comps == 0:
                comps = min(all_enc_train.shape[0], all_enc_train.shape[1])
            else:
                comps = self.pca_comps
            #        from zca import ZCA
            #        pca2 = ZCA()
            #        pca2.fit(all_enc_train)
            #        all_enc_train = pca2.transform(all_enc_train)
            pca2 = PCA(comps, whiten=True)
            all_enc_train = pca2.fit_transform(all_enc_train)

            print('perform pca')
            all_enc_test = pca2.transform(all_enc_test)

 #           print('> evaluate')
#            evaluate(all_enc_test, labels_test)

            all_enc_test = np.sign(all_enc_test) * np.sqrt(np.abs(all_enc_test))
            all_enc_test = normalize(all_enc_test)

            indices, dist_matrix = self.get_ranking(all_enc_train,all_enc_test)

            #print('> evaluate after power norm')
            #evaluate(all_enc_test, labels_test)

        if self.standardize:
            print('> standardize')
            scaler = StandardScaler()
            all_enc_train = scaler.fit_transform(all_enc_train)
            all_enc_test = scaler.fit_transform(all_enc_test)
            print('> evaluate')
            evaluate(all_enc_test, labels_test)

        if self.esvm:
            print('> esvm computation')
            # TODO: for a real setup you actually want to cross-validate C
            #        all_enc_test = esvm(all_enc_test, all_enc_train, args.C)
            indices, dist_matrix = self.get_ranking(all_enc_train,all_enc_test)

        return indices, dist_matrix
    def create_ranking(self,encs, dist_matrix=None):
        """
        evaluate encodings assuming using associated labels
        parameters:
            encs: TxK*D encoding matrix
            labels: array/list of T labels
        """
        if dist_matrix is None:
            dist_matrix = distances(encs)

        # mask out distance with itself
        #    np.fill_diagonal(dist_matrix, np.finfo(dist_matrix.dtype).max)
        print(dist_matrix)
        dist_matrix = dist_matrix[-1, :-1]
        # sort each row of the distance matrix
        indices = dist_matrix.argsort()

        return indices, dist_matrix

    def get_ranking(self,train_encoders,test_encodings):

#        all_enc_test = runF('encs_test_esvm.pkl.gz', True,
#                            self.tmp_folder,
#                            esvm, test_encodings, train_encoders, self.C)

        rank_indices, dist_matrix = self.create_ranking(test_encodings)#

        return rank_indices, dist_matrix



def get_data_from_dataloader(data_loader_file,train=True,gt=True,application=r"Testing/Dataloaders"):

    images = []
    labels = []

    style_samples = []
    style_labels = []

    print("application ",application, " config file ",data_loader_file)
    gdm = instantiate_completely(application,data_loader_file)
    if train:
        dl = gdm.train_dataloader()
    else:
        dl = gdm.test_dataloader()

    for batch in dl:
        sample = batch
        img = np.uint8(255* (1.0-(sample[IMAGE][0,0]+0.5)))
        images.append(img)
        labels.append(sample[WRITER])

        if not gt:
            style_labels.append(sample[WRITER])
            #TODO this should actuall go throughh the same transformations
            style_samples.append(sample[STYLE_SAMPLE][0,0])


    if not gt:
        images = images + style_samples
        labels = labels + style_labels


    return images, labels


def loadh5(h5_file,size=(640,768)):
    dataset = h5py.File(h5_file, "r")
    max_samples = len(dataset["images"])
    images = []
    labels = []

    for i in range(max_samples):

        labels.append(int(dataset["writer"][i].decode("utf-8")))
        images.append(dataset["images"][i].reshape(size))


    return images, labels




def parseArgs(parser):
    parser.add_argument('--tmp_folder', default=r"tmp",#'tmp',
                        help='default temporary folder')

    parser.add_argument('-str', '--suffix_train',
                        default='.png',
                        help='only chose those images with a specific suffix')
    parser.add_argument('-ste', '--suffix_test',
                        default='.jpg',
                        help='only chose those images with a specific suffix')
    parser.add_argument('--to_binary', action='store_true',
                        help='use OTSU binarization')
    parser.add_argument('--patch_size', type=int,
                        help='if given, we extract patches with this size'
                             'and use directly those as descriptors')
    parser.add_argument('--in_test',default="general768x768.yaml",    #"sdm640x768Home.yaml",
                        help='the input folder of the training images / features')
    parser.add_argument('--in_train',default="general768x768.yaml",
                        help='the input folder of the training images / features')
    parser.add_argument('--overwrite', action='store_true',default=True,
                        help='do not load pre-computed encodings')
    parser.add_argument('--powernorm', action='store_true',
                        help='use powernorm')
    parser.add_argument('--n_clusters', default=100, type=int,
                        help='number of clusters')
    parser.add_argument('--n_mvlad', default=1, type=int,
                        help='number of multi-vlad')
    parser.add_argument('--gmp', action='store_true',
                        help='use generalized max pooling')
    parser.add_argument('--gamma', default=1000, type=float,
                        help='regularization parameter of GMP')
    parser.add_argument('--esvm', default=True,action='store_true',
                        help='run esvm')
    parser.add_argument('--iesvm', action='store_true',
                        help='run intermediate esvm')
    parser.add_argument('--C', default=1000, type=float,
                        help='C parameter of the SVM')
    parser.add_argument('--pca_comps', default=-1, type=int,
                        help='use pca for the final descriptor')
    parser.add_argument('--ipca_comps', default=-1, type=int,
                        help='use pca on SIFT')
    parser.add_argument('--rerank', default=False,action='store_true',
                        help='sgr reranking')
    parser.add_argument('--standardize', action='store_true',
                        help='sgr reranking')
    return parser


def getFiles(folder, pattern, labelfile):
    """
    returns files and associated labels by reading the labelfile
    parameters:
        folder: inputfolder
        pattern: new suffix
        labelfiles: contains a list of filename and labels
    return: absolute filenames + labels
    """
    # read labelfile
    with open(labelfile, 'r') as f:
        all_lines = f.readlines()

    # get filenames from labelfile
    all_files = []
    labels = []
    check = True
    for line in all_lines:
        # using shlex we also allow spaces in filenames when escaped w. ""
        splits = shlex.split(line)
        file_name = splits[0]
        class_id = splits[1]

        # strip all known endings, note: os.path.splitext() doesnt work for
        # '.' in the filenames, so let's do it this way...
        for p in ['.pkl.gz', '.txt', '.png', '.jpg', '.tif', '.ocvmb', '.csv']:
            if file_name.endswith(p):
                file_name = file_name.replace(p, '')

        # get now new file name
        true_file_name = os.path.join(folder, file_name + pattern)
        all_files.append(true_file_name)
        labels.append(class_id)

    return all_files, labels


def computeKpts(img, sampling='keypoints', angle2zero=True):  # ersetzen durch superpoint
    """ compute the keypoints, i.e. where to extract the descriptors
    parameters:
        img: grayscale image
        sampling: canny, or keypoints
        angle2zero: only if keypoints -> set angle to 0
    """
    assert (img is not None)
    if sampling == 'keypoints':
        sift = cv2.SIFT_create()  # cv2.SIFT_create() instead of cv2.xfeatures2d.SIFT_create()
        keypoints = sift.detect(img, None)
        if angle2zero:
            for kp in keypoints:
                kp.angle = 0

        return keypoints

    if sampling == 'canny':
        keypoints = []
        edgeImg = cv2.Canny(img, 50, 200)
        for y in range(edgeImg.shape[0]):
            for x in range(edgeImg.shape[1]):
                if edgeImg[y, x] == 255:
                    keypoints.append(cv2.KeyPoint(x, y, 1))

        return keypoints

    return None


def computeSIFT(img, keypoints, norm_hellinger=True):
    """ compute SIFT at specific keypoints
    """
    sift = cv2.SIFT_create()  # cv2.SIFT_create() instead of cv2.xfeatures2d.SIFT_create()
    _, descriptors = sift.compute(img, keypoints)  # None anstatt norm_hellinger

    return descriptors


def loadRandomDescriptors(files, max_descriptors, patch_size):
    """
    compute roughly `max_descriptors` random local descriptors of dimension D from the files.
    parameters:
        files: list of image filenames
        max_descriptors: maximum number of descriptors (Q)
    returns: QxD matrix of descriptors
    """
    max_files = 747#300
    indices = np.random.permutation(max_files)


    files = np.array(files)[indices]

    # rough number of descriptors per file that we have to load
    # Note: actually we could also choose to use a lower filenumber
    # but in this way features from all files will be used
    max_descs_per_file = int(max_descriptors / len(files))

    descriptors = []
    for i in tqdm(range(len(files))):
        desc = computeDescs(files[i], True, True, patch_size)
        # get some random ones
        indices = np.random.choice(len(desc),
                                   min(len(desc),
                                       int(max_descs_per_file)),
                                   replace=False)
        desc = desc[:]#[indices]
        descriptors.append(desc)

    descriptors = np.concatenate(descriptors, axis=0)
    return descriptors


def dictionary(descriptors, n_clusters):
    """
    return cluster centers for the descriptors
    parameters:
        descriptors: NxD matrix of local descriptors
        n_clusters: number of clusters = K
    returns: KxD matrix of K clusters
    """

    cluster = MiniBatchKMeans(n_clusters,
                              compute_labels=False,
                              batch_size=100 * n_clusters).fit(descriptors)
    return cluster.cluster_centers_


def assignments(descriptors, clusters):
    """
    compute assignment matrix
    parameters:
        descriptors: TxD descriptor matrix
        clusters: KxD cluster matrix
    returns: TxK assignment matrix
    """
    # compute nearest neighbors
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.knnMatch(descriptors.astype(np.float32),
                               clusters.astype(np.float32),
                               k=1)

    # create hard assignment
    assignment = np.zeros((len(descriptors), len(clusters)))
    for e, m in enumerate(matches):
        assignment[e, m[0].trainIdx] = 1

    return assignment


def toBinary(mask):
    # test if not already binary
    if mask[mask == 255].sum() != np.sum(mask):
        # maybe binary between 0,1?
        if mask[mask == 1].sum() == mask.sum():
            mask *= 255
        else:  # make it binary
            ret, mask = cv2.threshold(mask, 125, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    return mask


def patchesFromKeypoints(img, keypoints, patchsize, ignore_border=True):
    # extract patch around keypoint
    all_patches = []
    # patch-size halves
    ps_h1 = patchsize / 2
    ps_h2 = patchsize - ps_h1
    for kpt in keypoints:
        if ignore_border:
            if kpt.pt[0] - ps_h1 < 0 or kpt.pt[1] - ps_h1 < 0 \
                    or kpt.pt[0] + ps_h2 > img.shape[1] \
                    or kpt.pt[1] + ps_h2 > img.shape[0]:
                continue

            patch = img[int(kpt.pt[1] - ps_h1): int(kpt.pt[1] + ps_h2), \
                    int(kpt.pt[0] - ps_h1): int(kpt.pt[0] + ps_h2)].copy()
        else:
            patch = img[max(0, int(kpt.pt[1] - ps_h1)):
                        min(img.shape[0], int(kpt.pt[1] + ps_h2)), \
                    max(0, int(kpt.pt[0] - ps_h1)):
                    max(img.shape[1], int(kpt.pt[0] + ps_h2))].copy()

        all_patches.append(patch.reshape(1, -1))

    return np.concatenate(all_patches, axis=0)


def computeDescs(fname, norm_hellinger=False, to_binary=False, patch_size=None):

    if not isinstance(fname,np.ndarray):
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    else:
        img =fname

    if img is None:
        raise IOError('cannot read', fname)
    if to_binary:
        img = toBinary(img)

    kpts = computeKpts(img, sampling='keypoints', angle2zero=True)
    if kpts is None or len(kpts) == 0:
        raise ValueError('cannot find any kpt for:', fname)
    if patch_size is not None:
        return patchesFromKeypoints(img, kpts, patch_size)

    descs = computeSIFT(img, kpts)
    #    kaze = cv2.KAZE_create()
    #    _, descs = kaze.detectAndCompute(img, None)
    if norm_hellinger:
        descs = normalize(descs, norm='l1')
        descs = np.sign(descs) * np.sqrt(np.abs(descs))

    return descs


def powernormalize(encs):
    encs = np.sign(encs) * np.sqrt(np.abs(encs))
    encs = normalize(encs, norm='l2')
    return encs


def vlad(files, mus, powernorm, gmp=False, gamma=1000, patch_size=None,
         pca=None, pca_v=None):
    """
    compute VLAD encoding for each files
    parameters:
        files: list of N files containing each T local descriptors of dimension
        D
        mus: KxD matrix of cluster centers
        gmp: if set to True use generalized max pooling instead of sum pooling
    returns: NxK*D matrix of encodings
    """
    K = mus.shape[0]
    encodings = []

    for f in tqdm(files):
        desc = computeDescs(f, True, True, patch_size)
        if pca is not None:
            desc = pca.transform(desc)
            desc = powernormalize(desc)
        a = assignments(desc, mus)

        T, D = desc.shape
        f_enc = np.zeros((D * K), dtype=np.float32)
        for k in range(mus.shape[0]):
            # it's faster to select only those descriptors that have
            # this cluster as nearest neighbor and then compute the
            # difference to the cluster center than computing the differences
            # first and then select

            # get only descriptors that are possible for this cluster
            nn = desc[a[:, k] > 0]
            # it can happen that we don't have any descriptors associated for
            # this cluster
            if len(nn) > 0:
                res = nn - mus[k]
                if gmp:
                    clf = Ridge(alpha=gamma,
                                fit_intercept=False,
                                solver='sparse_cg',
                                max_iter=500)  # conjugate gradient
                    clf.fit(res, np.ones((len(nn))))
                    f_enc[k * D:(k + 1) * D] = clf.coef_
                # compute residuals
                else:
                    f_enc[k * D:(k + 1) * D] = np.sum(res, axis=0)

        #        if pca_v is not None:
        #            f_enc = pca_v.transform(f_enc.reshape(1,-1))

        encodings.append(f_enc)

    encodings = np.vstack(encodings)

    # c) power normalization
    if powernorm:
        encodings = powernormalize(encodings)
    else:
        # l2 normalization
        encodings = normalize(encodings, norm='l2')

    return encodings


def distances(encs):
    """
    compute pairwise distances
    parameters:
        encs:  TxK*D encoding matrix
    returns: TxT distance matrix
    """
    # compute cosine distance = 1 - dot product between l2-normalized
    # encodings
    dists = 1.0 - encs.dot(encs.T)
    return dists


def evaluate(encs, labels, dist_matrix=None, rerank=True):
    evaluate_(encs, labels, dist_matrix)
#    print('> rerank')
#    _, dists_r = sgr.sgr_reranking(torch.tensor(encs), 2, 1,
#                                   float(0.4))
#    evaluate_(None, labels, dists_r.numpy())


def evaluate_single_sample_into_ranking(encs,style_sample_enc):
    encs_style = np.concatenate(list(encs,style_sample_enc),dim=0)
    dist_matrix = distances(encs_style)[-1,:-1]
    return dist_matrix.argsort()


def evaluate_(encs, labels, dist_matrix=None):
    """
    evaluate encodings assuming using associated labels
    parameters:
        encs: TxK*D encoding matrix
        labels: array/list of T labels
    """
    if dist_matrix is None:
        dist_matrix = distances(encs)

    # mask out distance with itself
    np.fill_diagonal(dist_matrix, np.finfo(dist_matrix.dtype).max)

    # sort each row of the distance matrix
    indices = dist_matrix.argsort()

    n_encs = len(dist_matrix)

    mAP = []
    correct = 0
   # computeStats(name="fk", dist_matrix=dist_matrix, labels_probe=labels)

    for r in range(n_encs):
        precisions = []
        rel = 0
        for k in range(n_encs - 1):
            if labels[indices[r, k]] == labels[r]:
                rel += 1
                precisions.append(rel / float(k + 1))
                if k == 0:
                    correct += 1
        avg_precision = np.mean(precisions)
        mAP.append(avg_precision)
    mAP = np.mean(mAP)

    print('Top-1 accuracy: {} - mAP: {}'.format(float(correct) / n_encs, mAP))


def dump(fname, obj):
    if fname.endswith('npy'):
        np.save(fname, obj.numpy())
    else:
        if not fname.endswith('.pkl.gz'):
            fname += '.pkl.gz'
        with gzip.open(fname, 'wb') as f_out:
            pickle.dump(obj, f_out)
    print('- dumped', fname)


def load(fname):
    # TODO: npy case
    if not fname.endswith('.pkl.gz'):
        fname += '.pkl.gz'
    with gzip.open(fname, 'rb') as f:
        mus = pickle.load(f)
    print('- loaded', fname)
    return mus


def runF(fname, overwrite, tmp_folder, func, *args, **kwargs):
    """
    before running a function it checks if we have already something saved
    """
    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder)

    path = os.path.join(tmp_folder, fname)
    if not os.path.exists(path) or overwrite:
        ret = func(*args, **kwargs)
        dump(path, ret)
    else:
        ret = load(path)

    return ret

def loop2(i,encs_test,to_classify,labels, same,C):
    esvm = LinearSVC(C=C, class_weight='balanced')#, dual='auto')
    to_classify[0] = encs_test[i]
    if same:
        esvm.fit(to_classify[np.arange(len(labels)) != (i + 1)],
                 labels[np.arange(len(labels)) != (i + 1)])
    else:
        esvm.fit(to_classify, labels)

    x = normalize(esvm.coef_, norm='l2')
    return x




def esvm(encs_test, encs_train, C=1000, same=False):
    """
    compute a new embedding using Exemplar Classification
    parameters:
        encs_test: NxD matrix
        encs_train: MxD matrix

    returns: new encs_test matrix (NxD)
    """

    # compute for each test encoding an E-SVM using the
    # encs as negatives
    labels = np.zeros(len(encs_train) + 1)
    labels[0] = 1

    to_classify = np.zeros((len(labels), encs_train.shape[1]),
                           dtype=encs_train.dtype)
    to_classify[1:] = encs_train

    def loop(i):
        esvm = LinearSVC(C=C, class_weight='balanced', dual='auto')
        to_classify[0] = encs_test[i]
        if same:
            esvm.fit(to_classify[np.arange(len(labels)) != (i + 1)],
                     labels[np.arange(len(labels)) != (i + 1)])
        else:
            esvm.fit(to_classify, labels)

        x = normalize(esvm.coef_, norm='l2')
        return x

   # new_encs = list(parmap(loop, tqdm(range(len(encs_test)))))
    #new_encs = np.zeros(encs_test.shape)

    new_encs = list()
    for i in range(len(encs_test)):
        #new_encs[i] = loop2(i,encs_test,to_classify,labels,same,C)
        new_encs.append(loop2(i,encs_test,to_classify,labels,same,C))


    #                           show_progress=True))
    #    new_encs = list(map(loop, tqdm(range(len(encs_test)))))
    new_encs = np.concatenate(new_encs, axis=0)
    # return new encodings
    return new_encs


def get_synthetical_data(file,size=(640,768)):
    dataset = h5py.File(file, "r")
    length = len(dataset["writer"])

    img_dict = dict()
    params_dict = dict()
    names = []
    for i in range(length):
        batch_size = 1
        if dataset.get("batch_size") is not None:
            batch_size = int(dataset["batch_size"][i].decode("utf-8"))

        # Go through images and add them to the dicionary based on the name
        images = dataset["images"][i].reshape(batch_size, 1, size[0], size[1])

        if dataset.get("name") is None:
            print("careful names and ids might swap")
            name = str(i)
        else:
            name = dataset["name"][i].decode("utf-8")

        if img_dict.get(name) is None:
            img_dict[name] = list()

        # So we add all the sampled images to the same batch
        for img in images:
            img_dict[name].append(img)

        if params_dict.get(name) is None:

            if dataset.get("style_sample") is not None:
                style_sample = dataset["style_sample"][i]
            else:
                style_sample = torch.zeros(size)

            params_dict[name] = {
                STYLE_SAMPLE: style_sample.reshape((1,size[0],size[1])),
                ORIGINAL: dataset["original"][i].reshape((1,size[0],size[1])),
                WRITER: int(dataset["writer"][i].decode("utf-8")),

                "name": name
            }
            names.append(name)
    return img_dict, params_dict,names

def computeStats(name, dist_matrix, labels_probe,
                 labels_gallery=None,
                 parallel=False, distance=True, nprocs=4,
                 eval_method='cosine', groups=None,save_dir=None):
    n_probe, n_gallery = dist_matrix.shape
    # often enough we make a leave-one-out-cross-validation
    # here we don't have a separation probe / gallery
    if labels_gallery is None:
        n_gallery -= 1
        labels_gallery = labels_probe
        # assert not needed or?
        assert(dist_matrix.shape[0] == dist_matrix.shape[1])
    assert(dist_matrix.shape[0] == len(labels_probe))
    assert(dist_matrix.shape[1] == len(labels_gallery))
    ind_probe = len(set(labels_probe))
    ind_gall = len(set(labels_gallery))
    labels_gallery = np.array(labels_gallery)
    labels_probe = np.array(labels_probe)
    print('number of probes: {}, individuals: {}'.format(n_probe, ind_probe))
    print('number of gallery: {}, individuals: {}'.format(n_gallery, ind_probe))
    if save_dir is not None:
        fp = open(save_dir,'w')
        fp.write('number of probes: {}, individuals: {}'.format(n_probe, ind_probe)+'\n')
        fp.write('number of gallery: {}, individuals: {}'.format(n_gallery, ind_probe)+'\n')

    indices = dist_matrix.argsort()

    if not distance:
        indices = indices[:, ::-1]

    # compute relevance list
    def loop_descr(r):
        rel_list = np.zeros((1, n_gallery))
        not_correct = []
        rel_cnt = 0
        for k in range(0, n_gallery):
            # if in the same group, then let's ignore it totally
            # -> not only ignore it but also don't forward the rank, i.e.
            # don't update the rel_list
            if groups is not None and groups[indices[r, k]] == groups[r]:
                continue
            else:
                if labels_gallery[indices[r, k]] == labels_probe[r]:
                    rel_list[0, rel_cnt] = 1
                elif k == 0:
                    not_correct.append((r, indices[r, k]))
                rel_cnt += 1
        return rel_list, not_correct

    # if parallel:
    #     all_rel, top1_fail = zip(*pc.parmap(loop_descr, range(n_probe), nprocs=nprocs))
    # else:
    all_rel, top1_fail = zip(*map(loop_descr, range(n_probe)))
        # make all computations with the rel-matrix
    rel_conc = np.concatenate(all_rel, 0)
    # are there any zero rows?
    z_rows = np.sum(rel_conc, 1)
    n_real2 = np.count_nonzero(z_rows)
    if n_real2 != rel_conc.shape[0]:
        print('WARNING: not for each query exist also a label in the gallery'
              '({} / {})!'.format(n_real2, rel_conc.shape[0]))
        if save_dir is not None:
            fp.write('WARNING: not for each query exist also a label in the gallery'
                  '({} / {})!'.format(n_real2, rel_conc.shape[0])+'\n')
    rel_mat = rel_conc[z_rows > 0]
    print('rel_mat.shape:', rel_mat.shape)
    if save_dir is not None:
        fp.write('rel_mat.shape: '+ str(rel_mat.shape) + '\n')

    prec_mat = np.zeros(rel_mat.shape)
    # TODO: this is still quite slow for large matrices
    # -> parallelize?
    soft2 = np.zeros(50)
    hard2 = np.zeros(4)
    for i in range(n_gallery):
        # TODO this could be done more efficiently
        # if we take the previous sum! -> cumsum
        rel_sum = np.sum(rel_mat[:, :i + 1], 1)
        # note: we could do this after the for loop (with correct shaped
        # i+1 vector: arange(1,n_gallery+1) and a rel_sum_mat
        # -> create rel_sum_mat
        prec_mat[:, i] = rel_sum / (i + 1)
        # could be partially speed up if outside w. rel_sum_mat
        if i < 20:
            soft2[i] = np.count_nonzero(rel_sum > 0) / float(n_real2)
        if i < 4:
            hh = rel_sum[np.isclose(rel_sum, (i + 1))]
            #            print 'i: {} len(hh): {}'.format(i, len(hh))
            hard2[i] = len(hh) / float(n_real2)
    print('correct: {} / {}'.format(np.sum(rel_mat[:, 0]), n_real2))
    print('top-k soft:', soft2[:10])
    print('top-k hard:', hard2)

    if save_dir is not None:
        fp.write('correct: {} / {}'.format(np.sum(rel_mat[:, 0]), n_real2) + '\n')
        fp.write('top-k soft: '+ str(soft2[:10]) + '\n')
        fp.write('top-k hard:'+ str(hard2) + '\n')


    # Average precisions
    ap = []
    for i in range(n_real2):
        ap.append(np.mean(prec_mat[i][rel_mat[i] == 1]))
    # mAP
    map1 = np.mean(ap)
    print('mAP mean(ap):', map1)
    if save_dir is not None:
        fp.write('mAP mean(ap): '+ str(map1) +'\n')

    # precision@x scores
    p2 = np.sum(prec_mat[:, 1]) / n_real2
    p3 = np.sum(prec_mat[:, 2]) / n_real2
    p4 = np.sum(prec_mat[:, 3]) / n_real2
    print('mean P@2,P@3,P@4:', p2, p3, p4)

    if save_dir is not None:
        fp.write('mean P@2,P@3,P@4:'+ str(p2)+" "+ str(p3) + " "+str(p4) +'\n')
        fp.close()




