import os
import numpy as np
from dataset import get_data, get_HOG, standardize
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    ######################## Get train/test dataset ########################
    X_train, X_test, Y_train, Y_test = get_data('dataset')
########################## Get HoG featues #############################
    H_train, H_test = get_HOG(X_train), get_HOG(X_test)
######################## standardize the HoG features ####################
    H_train, H_test = standardize(H_train), standardize(H_test)
########################################################################
######################## Implement you code here #######################
########################################################################
    # scikit-learn实现liner svm
    linear = SVC(kernel='linear', C=1)
    linear.fit(H_train, Y_train)

    # 输出分类正确率
    Y_pred = linear.predict(H_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print("test_accuracy: {:.2f}%".format(accuracy * 100))

    # 输出support vectors总数以及正负样本数
    n_sv = linear.n_support_
    print("support vectors:", np.sum(n_sv))
    sign = linear.dual_coef_[0]
    n_pos_sv = np.sum(sign > 0)
    n_neg_sv = np.sum(sign < 0)
    print("positive support vectors:", n_pos_sv)
    print("negative support vectors:", n_neg_sv)

    # 分别可视化正负样本α最大的20图
    selected_pos_sv = np.argsort(linear.dual_coef_[0])[::-1][:20]

    fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(32, 32))
    axs = axs.ravel()
    for i, idx in enumerate(selected_pos_sv):
        alpha = linear.dual_coef_[0][idx]
        img = X_train[linear.support_[idx]]
        axs[i].imshow(img)
        axs[i].axis('off')
        axs[i].set_title('α={:.2f}'.format(alpha))
    plt.show()

    selected_neg_sv = np.argsort(-linear.dual_coef_[0])[::-1][:20]

    fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(32, 32))
    axs = axs.ravel()
    for i, idx in enumerate(selected_neg_sv):
        alpha = linear.dual_coef_[0][idx]
        img = X_train[linear.support_[idx]]
        axs[i].imshow(img)
        axs[i].axis('off')
        axs[i].set_title('α={:.2f}'.format(alpha))
    plt.show()
