�
��0[c           @   sY  d  d l  Z  e  j d � d  d l j Z d  d l Z d  d l Z d  d l Z d  d l	 Z
 d  d l Z d  d l Z d  d l m Z d  d l Z d  d l Z d  d l Z d �  Z e d e j j d � Z d �  Z d �  Z d	 �  Z d
 �  Z d �  Z d �  Z d �  Z d �  Z d �  Z d �  Z  d �  Z! d �  Z" d �  Z# d �  Z$ d �  Z% d �  Z& d S(   i����Nt   Agg(   t   backendc         C   sf  d |  j  k r d  St j j t j � s; t j t j � n  t j j t j d j | � � } t	 j
 |  j  d � t	 j
 |  j  d � t	 j d � t	 j d � t	 j d � t	 j d d g d	 d
 �t	 j | d � t	 j �  t	 j
 |  j  d � t	 j
 |  j  d � t	 j d � t	 j d � t	 j d � t	 j d d g d	 d �t	 j | d � t	 j �  d  S(   Nt   accs	   {}-%s.jpgt   val_accs   model accuracyt   accuracyt   epocht   traint   testt   locs
   upper leftt   losst   val_losss
   model losss   upper right(   t   historyt   ost   patht   existst   configt	   plots_dirt   mkdirt   joint   formatt   pltt   plott   titlet   ylabelt   xlabelt   legendt   savefigt   close(   R   t   prefixt   img_path(    (    sG   /home/chenxingli/dengtaAI/keras-transfer-learning-for-oxford102/util.pyt   save_history   s*    !
s   Confusion matrixc   
      C   s�  d } t  j j | � s( t  j | � n  t j �  t j �  t j |  d d d | �t j | � t j	 �  t
 j t | � � } t j | | d d �t j | | � | r� |  j d � |  j d d	 � d
 d
 � t
 j f }  d GHn d GH|  GH|  j �  d } x t j t |  j d � t |  j d	 � � D]Q \ } }	 t j |	 | |  | |	 f d d d |  | |	 f | k rxd n d �q.Wt j �  t j d � t j d � | r�t j t  j j | d � � n t j t  j j | d � � d
 S(   s|   
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    s   ./confusion_matrix_plotst   interpolationt   nearestt   cmapt   rotationi-   t   floatt   axisi   Ns   Normalized confusion matrixs'   Confusion matrix, without normalizationg       @i    t   horizontalalignmentt   centert   colors   #BFD1D4t   blacks
   True labels   Predicted labels   normalized.jpgs   without_normalization.jpg(   R   R   R   R   R   t   clat   figuret   imshowR   t   colorbart   npt   aranget   lent   xtickst   ytickst   astypet   sumt   newaxist   maxt	   itertoolst   productt   ranget   shapet   textt   tight_layoutR   R   R   R   (
   t   cmt   classest	   normalizeR   R!   t   confusion_matrix_dirt
   tick_markst   thresht   it   j(    (    sG   /home/chenxingli/dengtaAI/keras-transfer-learning-for-oxford102/util.pyt   plot_confusion_matrix-   s6    


29'
c         C   sU   d d d d g } d } x6 | D]. } | t  t j t j j |  | � � � 7} q W| S(   Ns   *.pngs   *.jpgs   *.jpegs   *.bmpi    (   R/   t   globR   R   R   (   t   dir_patht   allowed_extensionst   numbert   e(    (    sG   /home/chenxingli/dengtaAI/keras-transfer-learning-for-oxford102/util.pyt   get_dir_imgs_numberW   s
    ,c          C   s�   d d d d h }  i d t  j 6d t  j 6} xm | D]e } t j | d � } xI | D]A } t j j | � \ } } | d |  k rS | | c d 7<qS qS Wq3 W| t  j t  _ | t  j t  _	 d S(	   sN   Walks through the train and valid directories
    and returns number of imagest   pngt   jpgt   jpegt   bmpi    s   **/*.*i   N(
   R   t	   train_dirt   validation_dirRE   t   iglobR   R   t   splitextt   nb_train_samplest   nb_validation_samples(   t   white_list_formatst	   dirs_infot   dt
   iglob_iterRB   t   filenamet   file_extension(    (    sG   /home/chenxingli/dengtaAI/keras-transfer-learning-for-oxford102/util.pyt   set_samples_info_   s    c         C   s�  d d d d h } t  �  } t g  t j |  � D]- } t j j t j j |  | � � r. | ^ q. � } d } x� | D]� } d | | <t j t j j |  | d � � } xI | D]A } t j j	 | � \ }	 }
 |
 d | k r� | | c d 7<q� q� W| d 7} qq Wt
 j t | j �  � � } t
 j t | j �  � � } d | t | � } | j �  } t  �  } xK | D]C } t j | | t | | � � } | d k r�| n d | | <qbW| S(	   NRK   RL   RM   RN   i    s   *.*i   g      �?(   t   dictt   sortedR   t   listdirR   t   isdirR   RE   RQ   RR   R-   R3   t   listt   valuesR5   R#   t   keyst   matht   log(   RW   RU   t   class_numbert   ot   dirst   kt
   class_nameRX   RB   t   _t   extt   totalt   max_samplest   muRb   t   class_weightt   keyt   score(    (    sG   /home/chenxingli/dengtaAI/keras-transfer-learning-for-oxford102/util.pyt   get_class_weighto   s*    	I
!	! c          C   sY   t  j }  t g  t j |  � D]- } t j j t j j |  | � � r | ^ q � t  _ d S(   s7   Returns classes based on directories in train directoryN(	   R   RO   R]   R   R^   R   R_   R   R=   (   RW   Rf   (    (    sG   /home/chenxingli/dengtaAI/keras-transfer-learning-for-oxford102/util.pyt   set_classes_from_train_dir�   s    	c             sK   d d l  m }  |  j �  d t �  � k r/ d S�  f d �  } | |  _ d S(   so   Overrides .next method of DirectoryIterator in Keras
      to reorder color channels for images from RGB to BGRi����(   t   DirectoryIteratort   custom_nextNc            sM   �  |  � \ } } | d  d  � d  d  d � d  d  � d  d  � f } | | f S(   Ni����(    (   t   selft   batch_xt   batch_y(   t   original_next(    sG   /home/chenxingli/dengtaAI/keras-transfer-learning-for-oxford102/util.pyRu   �   s    1(   t   keras.preprocessing.imageRt   t   nextt   str(   Rt   Ru   (    (   Ry   sG   /home/chenxingli/dengtaAI/keras-transfer-learning-for-oxford102/util.pyt&   override_keras_directory_iterator_next�   s    	c           C   s2   t  j r. t t t  j t t t  j � � � � Sd  S(   N(   R   R=   R\   t   zipR8   R/   t   None(    (    (    sG   /home/chenxingli/dengtaAI/keras-transfer-learning-for-oxford102/util.pyt   get_classes_in_keras_format�   s    	%c          O   s+   t  j d j t j � � } | j |  | �  S(   Ns	   models.{}(   t	   importlibt   import_moduleR   R   t   modelt
   inst_class(   t   argst   kwargst   module(    (    sG   /home/chenxingli/dengtaAI/keras-transfer-learning-for-oxford102/util.pyt   get_model_class_instance�   s    c         C   sA   |  j  d j t j �  g } |  j | � j g } t j | | � S(   Ni    (   t   layerst   inputt   Kt   learning_phaset	   get_layert   outputt   function(   t   mt   layert   xt   y(    (    sG   /home/chenxingli/dengtaAI/keras-transfer-learning-for-oxford102/util.pyt   get_activation_function�   s    c         C   s   |  | d g � } | d d S(   Ni    (    (   t   activation_functiont   X_batcht   activations(    (    sG   /home/chenxingli/dengtaAI/keras-transfer-learning-for-oxford102/util.pyt   get_activations�   s    c         C   s�   g  } g  } t  |  | � } x[ t t | � � D]G } t | | | g � }	 | j |	 � | j | | j d � d � q. Wt j | � }
 |
 j d d | � |
 j	 �  | d k r� |
 j
 t j d t d d d t �n |
 j
 t j d t �d  S(	   Nt   /i����i    t   classt   indext   modet   at   header(   R�   R8   R/   R�   t   appendt   splitt   pdt	   DataFramet   insertt   reset_indext   to_csvR   t   activations_patht   False(   R�   t   inputst   filesR�   t   batch_numbert   all_activationst   idst   afRB   t   actst
   submission(    (    sG   /home/chenxingli/dengtaAI/keras-transfer-learning-for-oxford102/util.pyt   save_activations�   s    "
%c          C   sZ   t  j j t j � r" t d � n  t t j d � �  }  |  j t t  j	 �  � � Wd  QXd  S(   Ns%   Previous process is not yet finished.t   w(
   R   R   R   R   t	   lock_filet   exitt   opent   writeR|   t   getpid(   R�   (    (    sG   /home/chenxingli/dengtaAI/keras-transfer-learning-for-oxford102/util.pyt   lock�   s    c           C   s,   t  j j t j � r( t  j t j � n  d  S(   N(   R   R   R   R   R�   t   remove(    (    (    sG   /home/chenxingli/dengtaAI/keras-transfer-learning-for-oxford102/util.pyt   unlock�   s    c           C   s   t  j j d � S(   Nt   2(   t   kerast   __version__t
   startswith(    (    (    sG   /home/chenxingli/dengtaAI/keras-transfer-learning-for-oxford102/util.pyt	   is_keras2�   s    c           C   s*   y t  j �  SWn t k
 r% t  j SXd  S(   N(   R�   R   t   AttributeErrort   _BACKEND(    (    (    sG   /home/chenxingli/dengtaAI/keras-transfer-learning-for-oxford102/util.pyt   get_keras_backend_name�   s    c          C   sT   d d  l  }  d d l m } |  j �  } t | j _ |  j d | � } | | � d  S(   Ni����(   t   set_sessionR   (   t
   tensorflowt    keras.backend.tensorflow_backendR�   t   ConfigProtot   Truet   gpu_optionst   allow_growtht   Session(   t   tfR�   t	   tf_configt   sess(    (    sG   /home/chenxingli/dengtaAI/keras-transfer-learning-for-oxford102/util.pyt   tf_allow_growth�   s    c           C   sw   y3 t  j �  d k r% t  j d � n t  j d � Wn= t k
 rr t  j d k rb t  j d � qs t  j d � n Xd  S(   Nt   theanot   channels_firstt   channels_lastt   thR�   (   R�   R   t   set_image_data_formatR�   R�   t   set_image_dim_ordering(    (    (    sG   /home/chenxingli/dengtaAI/keras-transfer-learning-for-oxford102/util.pyt   set_img_format�   s    ('   t
   matplotlibt   uset   matplotlib.pyplott   pyplotR   t   numpyR-   R   RE   t   pandasR�   R�   R�   R   R�   R   Rc   R6   R   R�   R<   t   BluesRD   RJ   R[   Rr   Rs   R}   R�   R�   R�   R�   R�   R�   R�   R�   R�   R�   R�   (    (    (    sG   /home/chenxingli/dengtaAI/keras-transfer-learning-for-oxford102/util.pyt   <module>   s<    	*															
