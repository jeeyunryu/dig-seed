import lmdb
import six
from PIL import Image

root = '/home/jyryu/workspace/DiG/dataset/Reann_MPSC/jyryu/lmdb/lmdb.test.wo.outs'
# index = 1


env = lmdb.open(root, max_readers=32, readonly=True)
txn = env.begin()

nSamples = int(txn.get(b"num-samples"))

for i in range(nSamples):
    img_key = b'image-%09d' % (i+1)
    imgbuf = txn.get(img_key)
    if imgbuf is None:
        import pdb;pdb.set_trace()
    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    try:
        img = Image.open(buf).convert('RGB')
    except IOError:
        print('Corrupted image for %d' % i+1)
        # return self[index + 1]
        raise ValueError('Corrupted image')