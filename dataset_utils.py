import numpy as np
import queue
import threading
import cv2


def parse_im_name(im_name, parse_type='id'):
  """Get the person id or cam from an image name."""
  assert parse_type in ('id', 'cam')
  if parse_type == 'id':
    parsed = int(im_name[:8])
  else:
    parsed = int(im_name[9:13])
  return parsed



class Counter(object):
  """A thread safe counter."""

  def __init__(self, val=0, max_val=0):
    self._value = val
    self.max_value = max_val
    self._lock = threading.Lock()

  def reset(self):
    with self._lock:
      self._value = 0

  def set_max_value(self, max_val):
    self.max_value = max_val

  def increment(self):
    with self._lock:
      if self._value < self.max_value:
        self._value += 1
        incremented = True
      else:
        incremented = False
      return incremented, self._value

  def get_value(self):
    with self._lock:
      return self._value

class Enqueuer(object):
  def __init__(self, get_element, num_elements, num_threads=1, queue_size=20):
    """
    Args:
      get_element: a function that takes a pointer and returns an element
      num_elements: total number of elements to put into the queue
      num_threads: num of parallel threads, >= 1
      queue_size: the maximum size of the queue. Set to some positive integer
        to save memory, otherwise, set to 0.
    """
    self.get_element = get_element
    assert num_threads > 0
    self.num_threads = num_threads
    self.queue_size = queue_size
    self.queue = queue.Queue(maxsize=queue_size)
    # The pointer shared by threads.
    self.ptr = Counter(max_val=num_elements)
    # The event to wake up threads, it's set at the beginning of an epoch.
    # It's cleared after an epoch is enqueued or when the states are reset.
    self.event = threading.Event()
    # To reset states.
    self.reset_event = threading.Event()
    # The event to terminate the threads.
    self.stop_event = threading.Event()
    self.threads = []
    for _ in range(num_threads):
      thread = threading.Thread(target=self.enqueue)
      # Set the thread in daemon mode, so that the main program ends normally.
      thread.daemon = True
      thread.start()
      self.threads.append(thread)

  def start_ep(self):
    """Start enqueuing an epoch."""
    self.event.set()

  def end_ep(self):
    """When all elements are enqueued, let threads sleep to save resources."""
    self.event.clear()
    self.ptr.reset()

  def reset(self):
    """Reset the threads, pointer and the queue to initial states. In common
    case, this will not be called."""
    self.reset_event.set()
    self.event.clear()
    # wait for threads to pause. This is not an absolutely safe way. The safer
    # way is to check some flag inside a thread, not implemented yet.
    time.sleep(5)
    self.reset_event.clear()
    self.ptr.reset()
    self.queue = Queue.Queue(maxsize=self.queue_size)

  def set_num_elements(self, num_elements):
    """Reset the max number of elements."""
    self.reset()
    self.ptr.set_max_value(num_elements)

  def stop(self):
    """Wait for threads to terminate."""
    self.stop_event.set()
    for thread in self.threads:
      thread.join()

  def enqueue(self):
    while not self.stop_event.isSet():
      # If the enqueuing event is not set, the thread just waits.
      if not self.event.wait(0.5): continue
      # Increment the counter to claim that this element has been enqueued by
      # this thread.
      incremented, ptr = self.ptr.increment()
      if incremented:
        element = self.get_element(ptr - 1)
        # When enqueuing, keep an eye on the stop and reset signal.
        while not self.stop_event.isSet() and not self.reset_event.isSet():
          try:
            # This operation will wait at most `timeout` for a free slot in
            # the queue to be available.
            self.queue.put(element, timeout=0.5)
            break
          except:
            pass
      else:
        self.end_ep()
    print('Exiting thread {}!!!!!!!!'.format(threading.current_thread().name))

class Prefetcher(object):
  """This helper class enables sample enqueuing and batch dequeuing, to speed
  up batch fetching. It abstracts away the enqueuing and dequeuing logic."""

  def __init__(self, get_sample, dataset_size, batch_size, final_batch=True,
               num_threads=1, prefetch_size=200):
    """
    Args:
      get_sample: a function that takes a pointer (index) and returns a sample
      dataset_size: total number of samples in the dataset
      final_batch: True or False, whether to keep or drop the final incomplete
        batch
      num_threads: num of parallel threads, >= 1
      prefetch_size: the maximum size of the queue. Set to some positive integer
        to save memory, otherwise, set to 0.
    """
    self.full_dataset_size = dataset_size
    self.final_batch = final_batch
    final_sz = self.full_dataset_size % batch_size
    if not final_batch:
      dataset_size = self.full_dataset_size - final_sz
    self.dataset_size = dataset_size
    self.batch_size = batch_size
    self.enqueuer = Enqueuer(get_element=get_sample, num_elements=dataset_size,
                             num_threads=num_threads, queue_size=prefetch_size)
    # The pointer indicating whether an epoch has been fetched from the queue
    self.ptr = 0
    self.ep_done = True

  def set_batch_size(self, batch_size):
    """You had better change batch size at the beginning of a new epoch."""
    final_sz = self.full_dataset_size % batch_size
    if not self.final_batch:
      self.dataset_size = self.full_dataset_size - final_sz
    self.enqueuer.set_num_elements(self.dataset_size)
    self.batch_size = batch_size
    self.ep_done = True

  def next_batch(self):
    """Return a batch of samples, meanwhile indicate whether the epoch is
    done. The purpose of this func is mainly to abstract away the loop and the
    boundary-checking logic.
    Returns:
      samples: a list of samples
      done: bool, whether the epoch is done
    """
    # Start enqueuing and other preparation at the beginning of an epoch.
    if self.ep_done:
      self.start_ep_prefetching()
    # Whether an epoch is done.
    self.ep_done = False
    samples = []
    for _ in range(self.batch_size):
      # Indeed, `>` will not occur.
      if self.ptr >= self.dataset_size:
        self.ep_done = True
        break
      else:
        self.ptr += 1
        sample = self.enqueuer.queue.get()
        # print('queue size {}'.format(self.enqueuer.queue.qsize()))
        samples.append(sample)
    # print 'queue size: {}'.format(self.enqueuer.queue.qsize())
    # Indeed, `>` will not occur.
    if self.ptr >= self.dataset_size:
      self.ep_done = True
    return samples, self.ep_done

  def start_ep_prefetching(self):
    """
    NOTE: Has to be called at the start of every epoch.
    """
    self.enqueuer.start_ep()
    self.ptr = 0

  def stop(self):
    """This can be called to stop threads, e.g. after finishing using the
    dataset, or when existing the python main program."""
    self.enqueuer.stop()
 
class PreProcessIm(object):
  def __init__(
      self,
      crop_prob=0,
      crop_ratio=1.0,
      resize_h_w=None,
      scale=True,
      im_mean=None,
      im_std=None,
      mirror_type=None,
      batch_dims='NCHW',
      prng=np.random):
    """
    Args:
      crop_prob: the probability of each image to go through cropping
      crop_ratio: a float. If == 1.0, no cropping.
      resize_h_w: (height, width) after resizing. If `None`, no resizing.
      scale: whether to scale the pixel value by 1/255
      im_mean: (Optionally) subtracting image mean; `None` or a tuple or list or
        numpy array with shape [3]
      im_std: (Optionally) divided by image std; `None` or a tuple or list or
        numpy array with shape [3]. Dividing is applied only when subtracting
        mean is applied.
      mirror_type: How image should be mirrored; one of
        [None, 'random', 'always']
      batch_dims: either 'NCHW' or 'NHWC'. 'N': batch size, 'C': num channels,
        'H': im height, 'W': im width. PyTorch uses 'NCHW', while TensorFlow
        uses 'NHWC'.
      prng: can be set to a numpy.random.RandomState object, in order to have
        random seed independent from the global one
    """
    self.crop_prob = crop_prob
    self.crop_ratio = crop_ratio
    self.resize_h_w = resize_h_w
    self.scale = scale
    self.im_mean = im_mean
    self.im_std = im_std
    self.check_mirror_type(mirror_type)
    self.mirror_type = mirror_type
    self.check_batch_dims(batch_dims)
    self.batch_dims = batch_dims
    self.prng = prng

  def __call__(self, im):
    return self.pre_process_im(im)


  @staticmethod
  def check_mirror_type(mirror_type):
    assert mirror_type in [None, 'random', 'always']

  @staticmethod
  def check_batch_dims(batch_dims):
    # 'N': batch size, 'C': num channels, 'H': im height, 'W': im width
    # PyTorch uses 'NCHW', while TensorFlow uses 'NHWC'.
    assert batch_dims in ['NCHW', 'NHWC']

  def set_mirror_type(self, mirror_type):
    self.check_mirror_type(mirror_type)
    self.mirror_type = mirror_type

  @staticmethod
  def rand_crop_im(im, new_size, prng=np.random):
    """Crop `im` to `new_size`: [new_w, new_h]."""
    if (new_size[0] == im.shape[1]) and (new_size[1] == im.shape[0]):
      return im
    h_start = prng.randint(0, im.shape[0] - new_size[1])
    w_start = prng.randint(0, im.shape[1] - new_size[0])
    im = np.copy(
      im[h_start: h_start + new_size[1], w_start: w_start + new_size[0], :])
    return im

  def pre_process_im(self, im):
    """Pre-process image.
    `im` is a numpy array with shape [H, W, 3], e.g. the result of
    matplotlib.pyplot.imread(some_im_path), or
    numpy.asarray(PIL.Image.open(some_im_path))."""

    # Randomly crop a sub-image.
    if ((self.crop_ratio < 1)
        and (self.crop_prob > 0)
        and (self.prng.uniform() < self.crop_prob)):
      h_ratio = self.prng.uniform(self.crop_ratio, 1)
      w_ratio = self.prng.uniform(self.crop_ratio, 1)
      crop_h = int(im.shape[0] * h_ratio)
      crop_w = int(im.shape[1] * w_ratio)
      im = self.rand_crop_im(im, (crop_w, crop_h), prng=self.prng)

    # Resize.
    if (self.resize_h_w is not None) \
        and (self.resize_h_w != (im.shape[0], im.shape[1])):
      im = cv2.resize(im, self.resize_h_w[::-1], interpolation=cv2.INTER_LINEAR)
    # scaled by 1/255.
    if self.scale:
      im = im / 255.

    # Subtract mean and scaled by std
    # im -= np.array(self.im_mean) # This causes an error:
    # Cannot cast ufunc subtract output from dtype('float64') to
    # dtype('uint8') with casting rule 'same_kind'
    if self.im_mean is not None:
      im = im - np.array(self.im_mean)
    if self.im_mean is not None and self.im_std is not None:
      im = im / np.array(self.im_std).astype(float)

    # May mirror image.
    mirrored = False
    if self.mirror_type == 'always' \
        or (self.mirror_type == 'random' and self.prng.uniform() > 0.5):
      im = im[:, ::-1, :]
      mirrored = True

    # The original image has dims 'HWC', transform it to 'CHW'.
    if self.batch_dims == 'NCHW':
      im = im.transpose(2, 0, 1)

    return im, mirrored
    
class PreProcessPartialIm(object):
  def __init__(
      self,
      crop_prob=0,
      crop_ratio=1.0,
      resize_h_w=None,
      scale=True,
      im_mean=None,
      im_std=None,
      mirror_type=None,
      batch_dims='NCHW',
      prng=np.random):
    """
    Args:
      crop_prob: the probability of each image to go through cropping
      crop_ratio: a float. If == 1.0, no cropping.
      resize_h_w: (height, width) after resizing. If `None`, no resizing.
      scale: whether to scale the pixel value by 1/255
      im_mean: (Optionally) subtracting image mean; `None` or a tuple or list or
        numpy array with shape [3]
      im_std: (Optionally) divided by image std; `None` or a tuple or list or
        numpy array with shape [3]. Dividing is applied only when subtracting
        mean is applied.
      mirror_type: How image should be mirrored; one of
        [None, 'random', 'always']
      batch_dims: either 'NCHW' or 'NHWC'. 'N': batch size, 'C': num channels,
        'H': im height, 'W': im width. PyTorch uses 'NCHW', while TensorFlow
        uses 'NHWC'.
      prng: can be set to a numpy.random.RandomState object, in order to have
        random seed independent from the global one
    """
    self.crop_prob = crop_prob
    self.crop_ratio = crop_ratio
    self.resize_h_w = resize_h_w
    self.scale = scale
    self.im_mean = im_mean
    self.im_std = im_std
    self.check_mirror_type(mirror_type)
    self.mirror_type = mirror_type
    self.check_batch_dims(batch_dims)
    self.batch_dims = batch_dims
    self.prng = prng

  def __call__(self, im):
    return self.pre_process_im1(im)


  @staticmethod
  def check_mirror_type(mirror_type):
    assert mirror_type in [None, 'random', 'always']

  @staticmethod
  def check_batch_dims(batch_dims):
    # 'N': batch size, 'C': num channels, 'H': im height, 'W': im width
    # PyTorch uses 'NCHW', while TensorFlow uses 'NHWC'.
    assert batch_dims in ['NCHW', 'NHWC']

  def set_mirror_type(self, mirror_type):
    self.check_mirror_type(mirror_type)
    self.mirror_type = mirror_type

  @staticmethod
  def rand_crop_im(im, new_size, prng=np.random):
    """Crop `im` to `new_size`: [new_w, new_h]."""
    if (new_size[0] == im.shape[1]) and (new_size[1] == im.shape[0]):
      return im
    h_start = prng.randint(0, im.shape[0] - new_size[1])
    w_start = prng.randint(0, im.shape[1] - new_size[0])
    im = np.copy(
      im[h_start: h_start + new_size[1], w_start: w_start + new_size[0], :])
    return im

  def pre_process_im1(self, im):
    """Pre-process image.
    `im` is a numpy array with shape [H, W, 3], e.g. the result of
    matplotlib.pyplot.imread(some_im_path), or
    numpy.asarray(PIL.Image.open(some_im_path))."""

    # Randomly crop a sub-image.
    if ((self.crop_ratio < 1)
        and (self.crop_prob > 0)
        and (self.prng.uniform() < self.crop_prob)):
      h_ratio = self.prng.uniform(self.crop_ratio, 1)
      w_ratio = self.prng.uniform(self.crop_ratio, 1)
      crop_h = int(im.shape[0] * h_ratio)
      crop_w = int(im.shape[1] * w_ratio)
      im = self.rand_crop_im(im, (crop_w, crop_h), prng=self.prng)

    # Resize.
    if (self.resize_h_w is not None) \
        and (self.resize_h_w != (im.shape[0], im.shape[1])):
      if im.shape[1]<=64:
        im = cv2.resize(im, (64, im.shape[0]), interpolation=cv2.INTER_LINEAR)
      else:
        im = cv2.resize(im, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_LINEAR)

    # scaled by 1/255.
    if self.scale:
      im = im / 255.

    # Subtract mean and scaled by std
    # im -= np.array(self.im_mean) # This causes an error:
    # Cannot cast ufunc subtract output from dtype('float64') to
    # dtype('uint8') with casting rule 'same_kind'
    if self.im_mean is not None:
      im = im - np.array(self.im_mean)
    if self.im_mean is not None and self.im_std is not None:
      im = im / np.array(self.im_std).astype(float)

    # May mirror image.
    mirrored = False
    if self.mirror_type == 'always' \
        or (self.mirror_type == 'random' and self.prng.uniform() > 0.5):
      im = im[:, ::-1, :]
      mirrored = True

    # The original image has dims 'HWC', transform it to 'CHW'.
    if self.batch_dims == 'NCHW':
      im = im.transpose(2, 0, 1)

    return im, mirrored