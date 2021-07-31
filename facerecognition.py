import face_recognition
# 导入图片
known_image_cyz = face_recognition.load_image_file("C:\\Users\\HYH\\Documents\\Program\\Python\\FaceRecognition\\FaceRecognition\\1.jpg")#图片路径
unknown_image = face_recognition.load_image_file("C:\\Users\\HYH\\Documents\\Program\\Python\\FaceRecognition\\FaceRecognition\\2.jpg")
# 编码获取128维特征向量
cyz_encoding = face_recognition.face_encodings(known_image_cyz)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
# 比较特征向量值，识别人脸
results = face_recognition.compare_faces([cyz_encoding], unknown_encoding, tolerance=0.5)
# 打印结果
print(results)
