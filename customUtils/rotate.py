# from PIL import Image

# # 이미지 열기
# img = Image.open('/home/jyryu/workspace/DiG/dataset/Reann_MPSC/jyryu/image/test/MPSC_img_41_9.jpg')

# # 왼쪽(반시계) 방향으로 90도 회전
# rotated_img = img.rotate(90, expand=True)

# # 결과 저장
# rotated_img.save("rotated_left.jpg")

# # 화면에 보여주기 (옵션)
# rotated_img.show()

from PIL import Image

# 이미지 열기
img = Image.open("/home/jyryu/workspace/DiG/dataset/Reann_MPSC/jyryu/image/test/MPSC_img_41_9.jpg")

# 오른쪽(시계 방향) 90도 회전
rotated_img = img.rotate(-90, expand=True)
# 또는
# rotated_img = img.rotate(270, expand=True)

# 결과 저장
rotated_img.save("rotated_right.jpg")

# 화면에 보여주기
rotated_img.show()
