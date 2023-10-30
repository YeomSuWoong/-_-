import cv2
import numpy as np
import matplotlib.pyplot as plt


# 1. choose two images
image_path1 = "/home/suwoong123/1_2_class/visod/midterm/3.jpg"
image_path2 = "/home/suwoong123/1_2_class/visod/midterm/4.jpg"


# 2. compute ORB keypoints and descriptors (opencv)
"""
def compute_orb_keypoints_and_descriptors(image_path):
    # 이미지 읽기
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # ORB 객체 생성
    orb = cv2.ORB_create()
    
    # 키포인트와 디스크립터 찾기
    keypoints, descriptors = orb.detectAndCompute(img, None)
    
    return keypoints, descriptors, img
"""
# SIFT 알고리즘을 이용한 feature extraction 수행
def compute_sift_keypoints_and_descriptors(image_path):
    # 이미지 읽기
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # SIFT 객체 생성
    sift = cv2.SIFT_create()
    
    # 키포인트와 디스크립터 찾기
    keypoints, descriptors = sift.detectAndCompute(img, None)
    
    return keypoints, descriptors, img


# 두 이미지에 대해 ORB 키포인트와 디스크립터 계산
#keypoints1, descriptors1, img1 = compute_orb_keypoints_and_descriptors(image_path2)
#keypoints2, descriptors2, img2 = compute_orb_keypoints_and_descriptors(image_path1)
keypoints1, descriptors1, img1 = compute_sift_keypoints_and_descriptors(image_path2)
keypoints2, descriptors2, img2 = compute_sift_keypoints_and_descriptors(image_path1)
# 결과 출력
img_keypoints1 = cv2.drawKeypoints(img1, keypoints1, outImage=None)
img_keypoints2 = cv2.drawKeypoints(img2, keypoints2, outImage=None)

plt.subplot(121)
plt.imshow(img_keypoints1, cmap='gray')
plt.title('Keypoints Image 1')

plt.subplot(122)
plt.imshow(img_keypoints2, cmap='gray')
plt.title('Keypoints Image 2')

plt.show()

"""
# 3. apply Bruteforce matching with Hamming distance (opencv)
def brute_force_matching(descriptors1, descriptors2):
    # Brute-Force 매처 객체를 생성합니다. 
    # cv2.NORM_HAMMING은 ORB 디스크립터에 적합한 거리 측정 방법입니다.
    # crossCheck=True는 양방향 매칭 결과만을 반환하도록 설정합니다.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # descriptors1의 각 디스크립터와 descriptors2의 각 디스크립터 사이의 거리를 계산하여 매칭을 수행합니다.
    matches = bf.match(descriptors1, descriptors2)
    
    # 매칭 결과를 거리에 따라 오름차순으로 정렬합니다. 
    # 이렇게 하면 가장 좋은 매칭 결과가 리스트의 앞쪽에 위치하게 됩니다.
    matches = sorted(matches, key=lambda x: x.distance)
    
    return matches  # 매칭 결과를 반환합니다.
"""
# flann 을 이용한 feature matching 수행
def flann_matching_sift(descriptors1, descriptors2):
    # FLANN 매개변수 설정
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # 검사할 이웃의 수
    
    # FLANN 매처 객체 생성
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # 매칭 수행
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    # 라티오 테스트를 사용하여 좋은 매칭을 필터링
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    return good_matches  # 좋은 매칭 결과를 반환합니다.

# 매칭 수행
#matches = brute_force_matching(descriptors1, descriptors2)
matches = flann_matching_sift(descriptors1, descriptors2)

# 매칭 결과 시각화
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:50], outImg=None)
plt.imshow(img_matches)
plt.title('Brute Force Matching')
plt.show()


# 4. implement RANSAC algorithm to compute homography matrix (DIY)

# RANSAC 을 이용하여 homography를 구하는 코드
def compute_homography_matrix(matches, keypoints1, keypoints2):
    # 매치된 키포인트에서 좌표 추출
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # RANSAC을 사용하여 호모그래피 매트릭스 계산
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    return H, mask
#"""
def compute_homography_matrix_(matches, keypoints1, keypoints2):
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # RANSAC, LMEDS 또는 RHO를 사용하여 호모그래피 매트릭스 계산
    #H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS)
    #H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RHO)
    
    return H, mask
#""" 

""" RANSAC 을 이용하여 homography를 구하는 코드 -> 추정 결과가 이상하여 기각!!
from skimage.transform import ProjectiveTransform, matrix_transform
def ransac_homography(matches, keypoints1, keypoints2, threshold=5, max_iterations=1000):
    best_H = None
    best_inliers = []

    for _ in range(max_iterations):
        # 4개의 랜덤한 매치 선택
        np.random.shuffle(matches)
        random_matches = matches[:4]

        src_pts = np.array([keypoints1[m.queryIdx].pt for m in random_matches])
        dst_pts = np.array([keypoints2[m.trainIdx].pt for m in random_matches])

        # 호모그래피 행렬 추정
        H = ProjectiveTransform()
        H.estimate(src_pts, dst_pts)

        inliers = []
        for match in matches:
            src_pt = np.array(keypoints1[match.queryIdx].pt)
            dst_pt = np.array(keypoints2[match.trainIdx].pt)

            # 변환 적용
            transformed_pt = matrix_transform([src_pt], H.params)
            
            # 오차 계산
            error = euclidean(transformed_pt[0], dst_pt)

            if error < threshold:
                inliers.append(match)

        # 최적의 모델 업데이트
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = H.params

    return best_H, best_inliers

# 앞서 정의된 다른 함수를 사용하여 매칭과 키포인트를 가져옵니다.

H, inliers = ransac_homography(matches, keypoints1, keypoints2)
print("Homography Matrix: \n", H)

"""

#H, inliers = ransac_homography(matches, keypoints1, keypoints2)
H, mask = compute_homography_matrix_(matches, keypoints1, keypoints2)
#H, mask = compute_homography_matrix_msac(matches, keypoints1, keypoints2)
print("Homography Matrix: \n", H)


# 5. prepare a panorama image using the homography matrix (DIY)
def prepare_panorama_size(images, H):
    # 각 이미지에 대한 좌표 구하기
    corners = []
    for img in images:
        h, w = img.shape[:2]
        corners.append([[0, 0], [0, h-1], [w-1, 0], [w-1, h-1]])

    # 두 번째 이미지의 좌표를 첫 번째 이미지 좌표계로 변환
    corners_transformed = cv2.perspectiveTransform(np.float32([corners[1]]), H)[0]

    # 변환된 좌표와 첫 번째 이미지의 좌표를 합침
    all_corners = np.concatenate((corners[0], corners_transformed), axis=0)

    # 새로운 이미지의 크기 계산
    x_min = min(all_corners[:, 0])
    x_max = max(all_corners[:, 0])
    y_min = min(all_corners[:, 1])
    y_max = max(all_corners[:, 1])

    width = round(x_max - x_min)
    height = round(y_max - y_min)

    # 변환 매트릭스 업데이트 (이동만 적용)
    translation = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])

    return (width, height), H_translation

# 이미지 로드
images = [cv2.imread("/home/suwoong123/1_2_class/visod/midterm/3.jpg"), cv2.imread("/home/suwoong123/1_2_class/visod/midterm/4.jpg")]

# 출력 이미지의 크기와 변환 매트릭스 계산
(output_width, output_height), H_translation = prepare_panorama_size(images, H)

# 6. warp two images to the panorama image using the homography matrix (DIY)
def create_panorama(images, H, output_size, H_translation):
    (output_width, output_height) = output_size
    
    # 첫 번째 이미지를 결과 이미지에 복사
    panorama = cv2.warpPerspective(images[0], H_translation, (output_width, output_height))
    
    # 호모그래피 매트릭스에 변환 매트릭스를 곱함
    H_total = np.dot(H_translation, H)
    
    # 두 번째 이미지를 워핑하여 결과 이미지에 추가
    panorama = cv2.warpPerspective(images[1], H_total, (output_width, output_height), panorama, borderMode=cv2.BORDER_TRANSPARENT)
    
    return panorama

# 파노라마 이미지 생성
panorama = create_panorama(images, H, (output_width, output_height), H_translation)

# Matplotlib을 사용하여 결과 이미지를 화면에 출력
plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))  # OpenCV는 BGR로 이미지를 로드하므로 RGB로 변환
plt.axis('off')  # 축 정보를 숨김
plt.title('Panorama')  # 이미지의 제목 설정
plt.show()  # 이미지를 화면에 표시