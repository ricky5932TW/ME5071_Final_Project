import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

class FindFeatures():

    def __init__(self, DirectoryPath, K):
        self.DirectoryPath, self.FileNum, self.K, self.P1 = DirectoryPath, len(os.listdir(DirectoryPath)), K, K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        self.points_3d, self.points_color = np.zeros((1, 3)), np.zeros((1, 3))

    def find_features(self):
        j, k = -1, -1
        for i in range(self.FileNum - 1):
            while True:
                if os.path.isfile(r"{0}\{1}.JPG".format(self.DirectoryPath, i)) and i > j: j = i; break
                # if os.path.isfile(r"{0}\frame_{1}.jpg".format(self.DirectoryPath, i)) and i > j: j = i; break
                else: i = i + 1
            while True:
                if os.path.isfile(r"{0}\{1}.JPG".format(self.DirectoryPath, i + 1)) and i + 1 > k: k = i + 1; break
                # if os.path.isfile(r"{0}\frame_{1}.jpg".format(self.DirectoryPath, i + 1)) and i + 1 > k: k = i + 1; break
                else: i = i + 1
            print(j)

            while True:
                img0, img1 = cv2.imread(r"{0}\{1}.JPG".format(self.DirectoryPath, j)), cv2.imread(r"{0}\{1}.JPG".format(self.DirectoryPath, k))
                # img0, img1 = cv2.imread(r"{0}\frame_{1}.jpg".format(self.DirectoryPath, j)), cv2.imread(r"{0}\frame_{1}.jpg".format(self.DirectoryPath, k))
                img0gray, img1gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

                sift = cv2.SIFT_create()
                kp0, des0 = sift.detectAndCompute(img0gray, None)
                kp1, des1 = sift.detectAndCompute(img1gray, None)

                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des0, des1, k=2)

                good_matches = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance: good_matches.append(m)

                # img_matches = cv2.drawMatches(img0, kp0, img1, kp1, good_matches, None, matchColor=(0, 100, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                # plt.imshow(img_matches[:, :, ::-1])
                # plt.show()

                pts0_, pts1_ = [], []

                for match in good_matches:
                    if match.queryIdx < len(kp0) and match.queryIdx < len(kp1):
                        pts0_.append(kp0[match.queryIdx].pt); pts1_.append(kp1[match.queryIdx].pt)

                E, mask = cv2.findEssentialMat(np.array(pts0_), np.array(pts1_), self.K, method=cv2.RANSAC, prob=0.999, threshold=0.4, mask=None)
                try:
                    pts0, pts1 = np.array(pts0_)[mask.ravel() == 1], np.array(pts1_)[mask.ravel() == 1]

                    _, R, t, mask = cv2.recoverPose(E, pts0, pts1, self.K)

                    P2 = self.K @ np.hstack((R, t))

                    pts0_hom = cv2.convertPointsToHomogeneous(pts0).reshape(-1, 3).T
                    pts1_hom = cv2.convertPointsToHomogeneous(pts1).reshape(-1, 3).T
                    points_4d_hom = cv2.triangulatePoints(self.P1, P2, pts0_hom[:2], pts1_hom[:2])
                    points_3d = cv2.convertPointsFromHomogeneous(points_4d_hom.T)

                    H = np.vstack((np.hstack((R, t)), [0, 0, 0, 1]))
                    points_3d_h = np.hstack((points_3d.reshape((points_3d.shape[0], 3)), np.ones((points_3d.shape[0], 1))))
                    points_3d_cam = (H @ points_3d_h.T).T[:, :3]
                    points_2d_corrected = cv2.projectPoints(points_3d_cam, cv2.Rodrigues(R)[0], t, self.K, np.array([0, 0, 0, 0, 0]).astype(float))[0].reshape(-1, 2)

                    new_M = cv2.findHomography(pts1, cv2.projectPoints(points_3d.reshape((points_3d.shape[0], 3)), cv2.Rodrigues(R)[0], t, self.K, np.array([0, 0, 0, 0]).astype(float))[0].reshape(-1, 2), cv2.RANSAC, 5.0)[0]
                    P2 = self.K @ new_M @ np.hstack((R, t))

                    points_4d_hom = cv2.triangulatePoints(self.P1, P2, pts0_hom[:2], pts1_hom[:2])
                    points_3d = cv2.convertPointsFromHomogeneous(points_4d_hom.T)

                    self.points_3d = np.append(self.points_3d, points_3d.reshape((points_3d.shape[0], 3)), axis=0)
                    self.points_color = np.append(self.points_color, img0[pts0[:, 1].astype(int), pts0[:, 0].astype(int)], axis=0)
                    # self.points_color = np.append(self.points_color, img0[np.array(pts0_)[:, 1].astype(int), np.array(pts0_)[:, 0].astype(int)], axis=0)
                    self.P1 = P2
                    break
                except:
                    while True:
                        if os.path.isfile(r"{0}\frame_{1}.jpg".format(self.DirectoryPath, i + 1)) and i + 1 > k: k = i + 1; break
                        else: i = i + 1
            j = k - 1

        self.points_color = self.points_color.astype(float) / 255.0
        self.points_3d, self.points_color = np.delete(self.points_3d, 0, 0), np.delete(self.points_color, 0, 0)
        return self.points_3d, self.points_color

                    # new_M = cv2.findHomography(pts1, cv2.projectPoints(points_3d.reshape((points_3d.shape[0], 3)), cv2.Rodrigues(R)[0], t, self.K, np.array([0, 0, 0, 0]).astype(float))[0].reshape(-1, 2), cv2.RANSAC, 5.0)[0]
                    # P2 = self.K @ new_M @ np.hstack((R, t))
                    # points_4d_hom = cv2.triangulatePoints(self.P1, P2, pts0_hom[:2], pts1_hom[:2])
                    # points_3d = cv2.convertPointsFromHomogeneous(points_4d_hom.T)
                    # test = cv2.projectPoints(points_3d.reshape((points_3d.shape[0], 3)), cv2.Rodrigues(new_M[:, 0:3])[0], new_M[:, 3].T, self.K, np.array([0, 0, 0, 0]).astype(float))[0].reshape(-1, 2)
                    # P2 = self.K @ cv2.findHomography(pts1, cv2.projectPoints(points_3d.reshape((points_3d.shape[0], 3)), cv2.Rodrigues(R)[0], t, self.K, np.array([0, 0, 0, 0]).astype(float))[0].reshape(-1, 2), cv2.RANSAC, 5.0)[0] @ np.hstack((R, t))
                    # pts0_hom = cv2.convertPointsToHomogeneous(np.array(pts0_)).reshape(-1, 3).T
                    # pts1_hom = cv2.convertPointsToHomogeneous(np.array(pts1_)).reshape(-1, 3).T