import numpy as np
import sklearn.cluster 
import scipy.stats

import warnings


def l_mb(shgo_res):
    x_vec    = np.array([ shgo_res[e_id]['x'][0] for e_id in shgo_res ])
    e_id_vec = np.array([e_id for e_id in shgo_res])
    
    
    zipped = list(zip(x_vec, e_id_vec))
    
    m = [(zipped[idx][1] - zipped[idx+1][1]) / (zipped[idx][0] - zipped[idx+1][0]) for idx in range(len(zipped)-1)]
    b = [ zipped[idx][1] - zipped[idx][0] * m[idx] for idx in range(len(m))]
    
    return np.array(m), np.array(b)


def normalize_vector(x : 'np.ndarray'):
    min_x = np.min(x)
    max_x = np.max(x)

    return (x - min_x) / (max_x - min_x)

def get_line_intersection(m0 : 'float', b0 : 'float', m1 : float, b1 : float) -> 'np.ndarray':
    x_s = (b1 - b0) / (m0 - m1)
    y_s = m0 * x_s + b0

    return np.array([x_s, y_s])

class wavelet_tracking_res():
    def __init__(self, shgo_res):
        self.tau  = np.array([shgo_res[e_id]['x'][0] for e_id in shgo_res])
        self.e_id = np.array([e_id for e_id in shgo_res])


class IP_by_electrode_pair_clusterV0():
    def __init__(self, m : 'np.ndarray'
                 , addon_cluster_parameter :'np.ndarray'
                 , b : 'np.ndarray'
                 , wavelet_parameters : 'dict'
                 , norm_tau : float = 0.02
                 , xi=0.02, max_eps = 0.2):
        self._m                       = m
        self._addon_cluster_parameter = addon_cluster_parameter
        self._b                       = b

        self._xi      = xi
        self._max_eps = max_eps

        self._norm_tau          = norm_tau
        self._N_electrode_pairs = len(self._m)

        self._wavelet_parameters = wavelet_parameters
        self.wtr = wavelet_tracking_res(wavelet_parameters)

    def cluster_lmb_OPTICS(self) -> 'np.ndarray':
        electrode_ids = self._wavelet_parameters.keys()
        X      = np.array([self._m, self._addon_cluster_parameter]).T

        X_norm = X.copy()

        X_norm[:, 0] = X[:, 0] / self._norm_tau
        X_norm[:, 1] = X[:, 1] / self._N_electrode_pairs

        electrode_pair_count = X_norm.shape[0]

        select_valid = ~((np.isinf(X_norm) | np.isnan(X_norm))[:,0] | (np.isinf(X_norm) | np.isnan(X_norm))[:,1] )
        X_subset_norm    = X_norm[select_valid, :]

        OPTICS_subset = sklearn.cluster.OPTICS(
                min_samples=3
                , xi= self._xi #0.02
                , cluster_method="dbscan"
                , max_eps= self._max_eps #0.2
                ).fit(X_subset_norm)

        labels_OPTICS = np.array([-1] * electrode_pair_count)

        labels_OPTICS[select_valid] = OPTICS_subset.labels_

        base_entry = np.array([[np.nan]* len(electrode_ids)] * electrode_pair_count)
        for idx_elecPair in range(electrode_pair_count):
            base_entry[idx_elecPair, idx_elecPair:idx_elecPair+2] = labels_OPTICS[idx_elecPair]

        self._base_entry    = base_entry
        self._X             = X
        self._X_valid       = select_valid
        self._X_subset_norm = X_subset_norm
        self._OPTICS_subset = OPTICS_subset

        return base_entry

    def split_clusters_by_sign(self):
        self._X_s_sel_neg = self._X_subset_norm[:,0] < 0 
        self._X_s_sel_pos = self._X_subset_norm[:,0] > 0 

        self._label_neg = self._OPTICS_subset.labels_[self._X_s_sel_neg]
        self._label_pos = self._OPTICS_subset.labels_[self._X_s_sel_pos]

        self._X_subset_norm_neg = self._X_subset_norm[self._X_s_sel_neg, :]
        self._X_subset_norm_pos = self._X_subset_norm[self._X_s_sel_pos, :]

    def get_biggest_cluster_per_split(self):
        self._biggest_clust_neg = self.get_biggest_cluster_in_split(self._label_neg, self._X_s_sel_neg)
        self._biggest_clust_pos = self.get_biggest_cluster_in_split(self._label_pos, self._X_s_sel_pos)

        if self._biggest_clust_neg is None or self._biggest_clust_pos is None:
            raise ValueError("Clustter electrode pair clusters where not found.")

    def get_biggest_cluster_in_split(self, label : 'np.ndarray', sel : 'np.ndarray'):
        lu = np.unique(label)

        lu = lu[lu > -1]

        if len(lu) < 1:
            return None

        rr = np.zeros_like(lu)

        for lu_idx in np.arange(len(lu)):
            rr[lu_idx] = np.sum((self._OPTICS_subset.labels_ == lu[lu_idx]) & sel)

        biggest_clust = lu[np.argmax(rr)]


        return biggest_clust

    def fit_lines_per_split(self):
        self._e_neg_cluster_selector = self.get_electrode_selector_in_split_by_label(
                  self._biggest_clust_neg
                , self._X_s_sel_neg).any(axis=0)
        self._e_pos_cluster_selector = self.get_electrode_selector_in_split_by_label(
                  self._biggest_clust_pos
                , self._X_s_sel_pos).any(axis=0)

        self._e_neg_c_lingress = scipy.stats.linregress(
                self.wtr.tau[self._e_neg_cluster_selector] 
                , self.wtr.e_id[self._e_neg_cluster_selector]
                )

        self._e_pos_c_lingress = scipy.stats.linregress(
                self.wtr.tau[self._e_pos_cluster_selector] 
                , self.wtr.e_id[self._e_pos_cluster_selector]
                )
        pass

    def get_electrode_selector_in_split_by_label(self, label, sel):
        
        N_elec_pairs = self._X.shape[0]
        big_selector = np.zeros((N_elec_pairs, N_elec_pairs+1)).astype(bool)

        for idx in range(N_elec_pairs):
            big_selector[idx, idx:idx+2] = True

        big_selector[~self._X_valid, :] = False

        label_select = (self._OPTICS_subset.labels_ == label) & sel

        a = big_selector[self._X_valid, :]
        a[~label_select, :] = False
        big_selector[self._X_valid, :] = a

        return big_selector


    def get_IP_by_lineInterception(self) -> 'np.ndarray':
        self.IP = get_line_intersection( m0= self._e_neg_c_lingress.slope
                              , b0 = self._e_neg_c_lingress.intercept
                              , m1= self._e_pos_c_lingress.slope
                              , b1 = self._e_pos_c_lingress.intercept
                              )
        return self.IP

    def exclude_cluster_that_are_in_both_signs(self):
        for l in np.unique(self._OPTICS_subset.labels_):
            l_selector = self._OPTICS_subset.labels_ == l
            sel_neg = self._X_subset_norm[l_selector,0] < 0 
            sel_pos = self._X_subset_norm[l_selector,0] > 0 
            if np.any(sel_neg) and np.any(sel_pos):
                self._OPTICS_subset.labels_[l_selector] = -1
                #import pdb; pdb.set_trace()



    def find_IP(self):
        self.cluster_lmb_OPTICS()
        self.exclude_cluster_that_are_in_both_signs()
        self.split_clusters_by_sign()
        self.get_biggest_cluster_per_split()
        self.fit_lines_per_split()
        return self.get_IP_by_lineInterception()


def calc_error_onlyXVary_equalSpacing(single_muscle_sim_res, pIP = None, e_id_vec = None):
    p_IP_mean = single_muscle_sim_res._mean_p_IP()
    
    #Warning("equal spaceing needed")
    if e_id_vec is None:
        e_id_vec = [e_id for e_id in single_muscle_sim_res.shgo_res]
    
    dd_elec_pos_x = single_muscle_sim_res.\
        electrode_meta.\
        loc[single_muscle_sim_res.electrode_ids[0]:single_muscle_sim_res.electrode_ids[-1]]
    """
    . . . . .
     . . . .
      . . .
    """
    y_0 = dd_elec_pos_x.iloc[1].x
    y_1 = dd_elec_pos_x.iloc[-2].x # wegen DD
    
    x_0 = e_id_vec[0]
    x_1 = e_id_vec[-1]

    m = (y_0 - y_1) / (x_0 - x_1) 
    b = y_0 - x_0 * m 

    get_elec_pos_x = lambda e_id : (m) * e_id + b
    #import pdb; pdb.set_trace()
    
    if pIP is None:
        pIP = single_muscle_sim_res.IP_by_e_pair_c.IP
    
    error_res = get_elec_pos_x(pIP[1]) - p_IP_mean[0,0]
    return error_res

class IP_by_electrode_pair_clusterV1():
    def __init__(self, m : 'np.ndarray'
                 , addon_cluster_parameter :'np.ndarray'
                 , b : 'np.ndarray'
                 , wavelet_parameters : 'dict'
                 , norm_tau : float = 0.02
                 , xi=0.02, max_eps = 0.2):
        self._m                       = m
        self._addon_cluster_parameter = addon_cluster_parameter
        self._b                       = b

        self._xi      = xi
        self._max_eps = max_eps

        self._norm_tau          = norm_tau
        self._N_electrode_pairs = len(self._m)

        self._wavelet_parameters = wavelet_parameters
        self.wtr = wavelet_tracking_res(wavelet_parameters)

    def cluster_lmb_OPTICS(self) -> 'np.ndarray':
        electrode_ids = self._wavelet_parameters.keys()
        X      = np.array([self._m, self._addon_cluster_parameter]).T

        X_norm = X.copy()

        X_norm[:, 0] = X[:, 0] / self._norm_tau
        X_norm[:, 1] = X[:, 1] / self._N_electrode_pairs

        electrode_pair_count = X_norm.shape[0]

        select_valid = ~((np.isinf(X_norm) | np.isnan(X_norm))[:,0] | (np.isinf(X_norm) | np.isnan(X_norm))[:,1] )
        X_subset_norm    = X_norm[select_valid, :]

        OPTICS_subset = sklearn.cluster.OPTICS(
                min_samples=3
                , xi= self._xi #0.02
                , cluster_method="dbscan"
                , max_eps= self._max_eps #0.2
                ).fit(X_subset_norm)

        labels_OPTICS = np.array([-1] * electrode_pair_count)

        labels_OPTICS[select_valid] = OPTICS_subset.labels_

        base_entry = np.array([[np.nan]* len(electrode_ids)] * electrode_pair_count)
        for idx_elecPair in range(electrode_pair_count):
            base_entry[idx_elecPair, idx_elecPair:idx_elecPair+2] = labels_OPTICS[idx_elecPair]

        self._base_entry    = base_entry
        self._X             = X
        self._X_valid       = select_valid
        self._X_subset_norm = X_subset_norm
        self._OPTICS_subset = OPTICS_subset

        return base_entry

    def split_clusters_by_sign(self):
        self._X_s_sel_neg = self._X_subset_norm[:,0] < 0 
        self._X_s_sel_pos = self._X_subset_norm[:,0] > 0 

        self._label_neg = self._OPTICS_subset.labels_[self._X_s_sel_neg]
        self._label_pos = self._OPTICS_subset.labels_[self._X_s_sel_pos]

        self._X_subset_norm_neg = self._X_subset_norm[self._X_s_sel_neg, :]
        self._X_subset_norm_pos = self._X_subset_norm[self._X_s_sel_pos, :]

    def get_biggest_cluster_per_split(self):
        self._biggest_clust_neg = self.get_biggest_cluster_in_split(self._label_neg, self._X_s_sel_neg)
        self._biggest_clust_pos = self.get_biggest_cluster_in_split(self._label_pos, self._X_s_sel_pos)

        if self._biggest_clust_neg is None or self._biggest_clust_pos is None:
            raise ValueError("Clustter electrode pair clusters where not found.")

    def get_biggest_cluster_in_split(self, label : 'np.ndarray', sel : 'np.ndarray'):
        lu = np.unique(label)

        lu = lu[lu > -1]

        if len(lu) < 1:
            return None

        rr = np.zeros_like(lu)

        for lu_idx in np.arange(len(lu)):
            #rr[lu_idx] = np.sum((self._OPTICS_subset.labels_ == lu[lu_idx]) & sel)
            rr[lu_idx] = np.sum(label == lu[lu_idx])

        biggest_clust = lu[np.argmax(rr)]


        return biggest_clust

    def fit_lines_per_cluster(self, _X_s_selector, labels_subset):
        fit_dict  = {}

        best_fit_label = -1
        best_fit_R2    =  0

        best_electrode_sel = None

        for label in np.unique(labels_subset):
            if label < 0:
                continue

            electrode_sel = self.get_electrode_selector_in_split_by_label(label, _X_s_selector)\
                    .any(axis=0)

            fit_dict[label] = scipy.stats.linregress(
                    self.wtr.tau[electrode_sel]
                    , self.wtr.e_id[electrode_sel])

            current_R2 = (fit_dict[label].rvalue)**2 * np.sum(electrode_sel)

            if best_fit_R2 < current_R2:
                best_fit_R2        = current_R2
                best_fit_label     = label
                best_electrode_sel = electrode_sel

        return fit_dict[best_fit_label], best_fit_label, best_electrode_sel

    def set_lines_per_best_cluster_fit(self):
        best_fit_neg , best_fit_label_neg, best_electrode_sel_neg = self.fit_lines_per_cluster(
                self._X_s_sel_neg, self._label_neg)
        best_fit_pos , best_fit_label_pos, best_electrode_sel_pos = self.fit_lines_per_cluster(
                self._X_s_sel_pos, self._label_pos)

        if best_fit_label_neg >= 0:
            self._e_neg_c_lingress  = best_fit_neg
            self._biggest_clust_neg = best_fit_label_neg
            self._e_neg_cluster_selector = best_electrode_sel_neg

        if best_fit_label_pos >= 0:
            self._e_pos_c_lingress  = best_fit_pos
            self._biggest_clust_pos = best_fit_label_pos
            self._e_pos_cluster_selector = best_electrode_sel_pos


    def fit_lines_per_split(self):
        self._e_neg_cluster_selector = self.get_electrode_selector_in_split_by_label(
                  self._biggest_clust_neg
                , self._X_s_sel_neg).any(axis=0)
        self._e_pos_cluster_selector = self.get_electrode_selector_in_split_by_label(
                  self._biggest_clust_pos
                , self._X_s_sel_pos).any(axis=0)

        self._e_neg_c_lingress = scipy.stats.linregress(
                self.wtr.tau[self._e_neg_cluster_selector] 
                , self.wtr.e_id[self._e_neg_cluster_selector]
                )

        self._e_pos_c_lingress = scipy.stats.linregress(
                self.wtr.tau[self._e_pos_cluster_selector] 
                , self.wtr.e_id[self._e_pos_cluster_selector]
                )
        pass

    def get_electrode_selector_in_split_by_label(self, label, sel):
        
        N_elec_pairs = self._X.shape[0]
        big_selector = np.zeros((N_elec_pairs, N_elec_pairs+1)).astype(bool)

        for idx in range(N_elec_pairs):
            big_selector[idx, idx:idx+2] = True

        big_selector[~self._X_valid, :] = False

        label_select = (self._OPTICS_subset.labels_ == label) & sel

        a = big_selector[self._X_valid, :]
        a[~label_select, :] = False
        big_selector[self._X_valid, :] = a

        return big_selector


    def get_IP_by_lineInterception(self) -> 'np.ndarray':
        self.IP = get_line_intersection( m0= self._e_neg_c_lingress.slope
                              , b0 = self._e_neg_c_lingress.intercept
                              , m1= self._e_pos_c_lingress.slope
                              , b1 = self._e_pos_c_lingress.intercept
                              )
        return self.IP

    def exclude_cluster_that_are_in_both_signs(self):
        for l in np.unique(self._OPTICS_subset.labels_):
            l_selector = self._OPTICS_subset.labels_ == l
            sel_neg = self._X_subset_norm[l_selector,0] < 0 
            sel_pos = self._X_subset_norm[l_selector,0] > 0 
            if np.any(sel_neg) and np.any(sel_pos):
                self._OPTICS_subset.labels_[l_selector] = -1
                #import pdb; pdb.set_trace()



    def find_IP(self):
        self.cluster_lmb_OPTICS()
        self.exclude_cluster_that_are_in_both_signs()
        self.split_clusters_by_sign()

        self.set_lines_per_best_cluster_fit()

        return self.get_IP_by_lineInterception()









class IP_by_line_cluster():
    def __init__(self, 
                 m : 'np.ndarray', b: 'np.ndarray'
                 , eps= 0.12, min_samples = 3
                 , norm_tau = 0.02
                 , norm_id  = None
                 ) -> 'IP_by_line_cluster':
        self._m = m
        self._b = b

        self._eps         = eps
        self._min_samples = min_samples

        self._norm_tau = norm_tau
        self._norm_id  = norm_id


    def calc_line_intersection(self): 
        self._N_electrodes = len(self._m)

        X = np.array([self._m, self._b]).T

        sel_m_neg = self._m < 0
        sel_m_pos = self._m > 0

        N = np.sum(sel_m_neg) * np.sum(sel_m_pos)

        self._line_intersections = np.zeros((N,2))

        n = 0

        for mb_neg in X[sel_m_neg]:
            for mb_pos in X[sel_m_pos]:
                self._line_intersections[n,:] = get_line_intersection( m0 = mb_neg[0], b0 = mb_neg[1], \
                        m1 = mb_pos[0], b1 = mb_pos[1])
                n += 1

    def filter_for_valid_intersections(self):
        line_intersections_sel_valid_row = ~np.any(np.isnan(self._line_intersections), axis = 1)
        line_intersections_subset = self._line_intersections[line_intersections_sel_valid_row, :]

        self._line_intersections_subset = line_intersections_subset

        line_intersections_subset_norm = line_intersections_subset.copy()
        line_intersections_subset_norm[:, 0] = line_intersections_subset[:, 0] / self._norm_tau
        if self._norm_id is None:
            line_intersections_subset_norm[:, 1] = line_intersections_subset[:, 1] / self._N_electrodes
        else:
            line_intersections_subset_norm[:, 1] = line_intersections_subset[:, 1] / self._norm_id

        self._line_intersections_subset_norm = line_intersections_subset_norm

    def doCluster(self):
        dbscan = sklearn.cluster.DBSCAN(eps=self._eps, min_samples=self._min_samples)

        self._dbscan_res = dbscan.fit(self._line_intersections_subset_norm)

    def calc_biggest_cluster_center(self):
        if len(self._dbscan_res.labels_) <= 1:
            return None

        lu = np.unique(self._dbscan_res.labels_)
        rr = np.zeros_like(lu)

        for lu_idx in np.arange(len(lu)):
            rr[lu_idx] = np.sum(self._dbscan_res.labels_ == lu[lu_idx])

        biggest_clust = lu[np.argmax(rr)]

        if biggest_clust == -1:
            #warnings.warn("biggest cluster was outliers")
            rr[lu == -1] = 0
            biggest_clust = lu[np.argmax(rr)]
            #return None

        sel_winning_clust = self._dbscan_res.labels_ == biggest_clust

        x = np.mean(self._line_intersections_subset_norm[sel_winning_clust, 0])
        y = np.mean(self._line_intersections_subset_norm[sel_winning_clust, 1])

        self._biggest_clust     = biggest_clust
        self._mean_biggest_clust_norm = np.array([x,y])

        x = np.mean(self._line_intersections_subset[sel_winning_clust, 0])
        y = np.mean(self._line_intersections_subset[sel_winning_clust, 1])

        x_std = np.std(self._line_intersections_subset[sel_winning_clust, 0])
        y_std = np.std(self._line_intersections_subset[sel_winning_clust, 1])


        self._mean_biggest_clust = np.array([x,y])
        self._std_biggest_clust  = np.array([x_std, y_std])

    def find_IP(self):
        self.calc_line_intersection()
        self.filter_for_valid_intersections()
        
        if (self._line_intersections_subset_norm is None):
            return None
        if (self._line_intersections_subset_norm.shape[0] < 1):
            return None
                
        
        self.doCluster()
        self.calc_biggest_cluster_center()

        return self._mean_biggest_clust


IP_by_electrode_pair_cluster = IP_by_electrode_pair_clusterV1
