import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
import matplotlib.image as mpimg
import itertools
import time

# 设置随机种子以确保可重复性
np.random.seed(42)


def load_mnist_data(train_path, test_path):
    """
    加载MNIST数据集
    """
    # 训练数据
    file_ls = os.listdir(train_path)
    data = np.zeros((60000, 784), dtype=float)
    label = np.zeros((60000,), dtype=int)
    flag = 0
    for dir_name in file_ls:
        files = os.listdir(os.path.join(train_path, dir_name))
        for file in files:
            filename = os.path.join(train_path, dir_name, file)
            img = mpimg.imread(filename)
            data[flag, :] = np.reshape(img, -1) / 255.0
            label[flag] = int(dir_name)
            flag += 1

    # 测试数据
    file_ls = os.listdir(test_path)
    xTesting = np.zeros((10000, 784), dtype=float)
    yTesting = np.zeros((10000,), dtype=int)
    flag = 0
    for dir_name in file_ls:
        files = os.listdir(os.path.join(test_path, dir_name))
        for file in files:
            filename = os.path.join(test_path, dir_name, file)
            img = mpimg.imread(filename)
            xTesting[flag, :] = np.reshape(img, -1) / 255.0
            yTesting[flag] = int(dir_name)
            flag += 1

    return data, label, xTesting, yTesting


def preprocess_data(x_train, y_train, x_test, y_test, validation_size=0.05):
    """
    数据预处理和划分
    """
    # 划分训练集和验证集
    xTraining, xValidation, yTraining, yValidation = train_test_split(
        x_train, y_train, test_size=validation_size, random_state=42, stratify=y_train)

    return xTraining, yTraining, xValidation, yValidation, x_test, y_test


def train_svm_model(x_train, y_train, kernel='rbf', C=10, gamma=0.01):
    """
    使用scikit-learn训练SVM模型
    """
    print("开始训练SVM模型...")
    start_time = time.time()

    # 创建SVM分类器
    svm_classifier = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42)

    # 训练模型
    svm_classifier.fit(x_train, y_train)

    end_time = time.time()
    print(f"训练完成，耗时: {end_time - start_time:.2f} 秒")

    return svm_classifier


def evaluate_model(model, x_test, y_test):
    """
    评估模型性能
    """
    # 预测标签和概率
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)

    accuracy = accuracy_score(y_test, y_pred)

    return accuracy, y_pred, y_prob


def main():
    """
    主函数
    """
    # 数据路径 - 请根据实际情况修改
    train_path = 'E:\\课程作业\\机器学习\\手写字体识别\\MNIST\\train'
    test_path = 'E:\\课程作业\\机器学习\\手写字体识别\\MNIST\\test'

    print("正在加载MNIST数据集...")
    x_train, y_train, x_test, y_test = load_mnist_data(train_path, test_path)

    print("数据预处理...")
    xTraining, yTraining, xValidation, yValidation, xTesting, yTesting = preprocess_data(
        x_train, y_train, x_test, y_test, validation_size=0.05)

    print(f"训练集大小: {xTraining.shape[0]}")
    print(f"验证集大小: {xValidation.shape[0]}")
    print(f"测试集大小: {xTesting.shape[0]}")

    # 训练SVM模型
    # 使用RBF核函数，经过调优的参数
    svm_model = train_svm_model(xTraining, yTraining, kernel='rbf', C=10, gamma=0.01)

    print("在验证集上评估模型...")
    val_accuracy, val_p_label, val_p_val = evaluate_model(svm_model, xValidation, yValidation)
    print(f"验证集准确率: {val_accuracy:.4f}")

    print("在测试集上评估模型...")
    test_accuracy, test_p_label, test_p_val = evaluate_model(svm_model, xTesting, yTesting)
    print(f"测试集准确率: {test_accuracy:.4f}")

    # 计算混淆矩阵
    cm = confusion_matrix(yTesting, test_p_label)

    # 生成分类报告
    class_report = classification_report(yTesting, test_p_label)
    print("\n分类报告:")
    print(class_report)

    # 可视化结果 - 分别展示四张子图
    print("生成可视化结果...")

    # 1. 标准混淆矩阵
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=14)
    plt.colorbar()
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # 在混淆矩阵中添加数值
    threshold = cm.max() / 2
    for i in range(10):
        for j in range(10):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black")

    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show(block=False)
    print("标准混淆矩阵已生成并显示")

    # 2. 归一化混淆矩阵
    plt.figure(figsize=(8, 6))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm_normalized, cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix', fontsize=14)
    plt.colorbar()
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # 在归一化混淆矩阵中添加数值
    for i in range(10):
        for j in range(10):
            plt.text(j, i, format(cm_normalized[i, j], '.2f'),
                     horizontalalignment="center",
                     color="white" if cm_normalized[i, j] > 0.5 else "black")

    plt.tight_layout()
    plt.savefig('normalized_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show(block=False)
    print("归一化混淆矩阵已生成并显示")

    # 3. ROC曲线
    plt.figure(figsize=(8, 6))

    if test_p_val is not None:
        # 将标签二值化用于ROC曲线
        y_test_bin = label_binarize(yTesting, classes=list(range(10)))

        # 计算每个类别的ROC曲线和AUC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        # 只计算几个代表性类别的ROC曲线，避免计算所有10个类别
        representative_classes = [0, 1, 2, 9]  # 选择几个代表性类别

        for i in representative_classes:
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], test_p_val[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], lw=2,
                     label='Class {0} (AUC = {1:0.4f})'.format(i, roc_auc[i]))

        # 计算微平均ROC曲线
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), test_p_val.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        plt.plot(fpr["micro"], tpr["micro"],
                 label='Micro-average (AUC = {0:0.4f})'.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'ROC Curve Not Available\n(Model does not provide probability prediction)',
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=14)
        plt.title('ROC Curves Not Available', fontsize=14)

    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=150, bbox_inches='tight')
    plt.show(block=False)
    print("ROC曲线已生成并显示")

    # 4. 准确率和性能指标总结
    plt.figure(figsize=(8, 6))
    plt.axis('off')

    # 计算每个类别的准确率
    class_accuracies = []
    for i in range(10):
        class_mask = yTesting == i
        if np.sum(class_mask) > 0:
            class_accuracy = accuracy_score(yTesting[class_mask], test_p_label[class_mask])
            class_accuracies.append((i, class_accuracy))

    # 创建总结文本
    summary_text = f"SVM Model Performance Summary on MNIST Dataset:\n\n"
    summary_text += f"Overall Accuracy: {test_accuracy:.4f}\n\n"
    summary_text += "Per-Class Accuracy:\n"

    for i, acc in class_accuracies:
        summary_text += f"  Class {i}: {acc:.4f}\n"

    # 添加精确率、召回率和F1分数的平均值
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(yTesting, test_p_label, average='macro')
    summary_text += f"\nMacro-average:\n"
    summary_text += f"  Precision: {precision:.4f}\n"
    summary_text += f"  Recall: {recall:.4f}\n"
    summary_text += f"  F1-score: {f1:.4f}\n"

    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.title('Performance Summary', fontsize=14)

    plt.tight_layout()
    plt.savefig('performance_summary.png', dpi=150, bbox_inches='tight')
    plt.show(block=False)
    print("性能总结已生成并显示")

    print("可视化完成！")
    print(f"最终测试准确率: {test_accuracy:.4f}")

    # 保存结果
    results = {
        'test_accuracy': test_accuracy,
        'confusion_matrix': cm,
        'predictions': test_p_label,
        'true_labels': yTesting
    }

    return svm_model, results


if __name__ == "__main__":
    model, results = main()