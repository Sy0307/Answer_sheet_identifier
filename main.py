import cv2
import numpy as np
from itertools import combinations
import pytesseract

from itertools import combinations
from tkinter import Tk, Label, Button, filedialog, messagebox, Text , Checkbutton , BooleanVar , Entry
from PIL import Image, ImageTk


# 计算两条直线的交点
def compute_intersection(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    return int(np.round(x0)), int(np.round(y0))


def student_id_get(image):
    # 将图像转换为灰度图
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用自适应阈值进行二值化处理
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # 查找所有轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 再次处理裁剪后的图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # 查找新的轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 过滤出数字区域
    digit_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = h / w
        if 1.5 < aspect_ratio < 3.0 and 10 < w < 50 and 10 < h < 40:
            digit_contours.append((x, y, w, h))

    # 按x排序，以从左到右的顺序读取数字
    digit_contours = sorted(digit_contours, key=lambda x: x[0])

    # 用于保存识别出的数字
    digits = []

    # 创建保存裁剪图像的文件夹
    # output_dir = 'digit_crops'
    # os.makedirs(output_dir, exist_ok=True)

    for i, (x, y, w, h) in enumerate(digit_contours):
        # 裁剪出每个数字区域
        digit = image[(int)(y-h*0.1) : (int)(y + h*1.1), (int)(x - w*0.1) : (int)(x + w*1.1)]
        # 展示裁剪出的数字区域
        # cv2.imshow(f"Digit {i}", digit)
        # cv2.waitKey(0)

        # 保存裁剪出的数字区域（可选）
        # crop_path = os.path.join(output_dir, f"digit_{i}.png")
        # cv2.imwrite(crop_path, digit)

        # 使用 pytesseract 识别裁剪出的数字
        # 配置参数以提高识别效果，只识别数字
        binary_digit = cv2.adaptiveThreshold(
            cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2,
        )
        
        
        config = r"--psm 9"
        recognized_digit = pytesseract.image_to_string(digit, lang="eng", config=config)
        print(f"Recognized digit: {recognized_digit}")

        recognized_digit = recognized_digit[0]
        # 添加到结果列表
        if(len(recognized_digit)==1) and recognized_digit.isdigit():
            digits.append(recognized_digit)

    # 拼接识别的所有数字，组成最终的学号
    student_id = "".join(digits)
    print(f"Student ID: {student_id}")
    return student_id


# 透视变换函数
def perspective_transform(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (1, 5), 1)  # 使用高斯模糊平滑图像
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # 使用 Canny 边缘检测
    edges = cv2.Canny(binary, 1000, 150)

    # 使用霍夫变换检测直线
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    # 如果检测到的直线数量不足，直接返回原图
    if lines is None or len(lines) < 4:
        print("无法检测到足够的直线")
        return image

    # 计算所有直线组合的交点
    lines = lines[:, 0, :]  # 提取线条参数
    intersections = []
    for line1, line2 in combinations(lines, 2):
        try:
            intersection = compute_intersection(line1, line2)
            intersections.append(intersection)
        except np.linalg.LinAlgError:
            continue

    # 保留图像范围内的交点
    intersections = [
        pt
        for pt in intersections
        if 0 <= pt[0] < image.shape[1] and 0 <= pt[1] < image.shape[0]
    ]

    # 如果交点不足4个，直接返回原图
    if len(intersections) < 4:
        print("无法检测到足够的交点")
        return image

    # 对交点进行排序并选取最接近四角的点
    intersections = sorted(intersections, key=lambda x: x[0] + x[1])
    tl = min(intersections, key=lambda p: p[0] + p[1])
    br = max(intersections, key=lambda p: p[0] + p[1])
    tr = max(intersections, key=lambda p: p[0] - p[1])
    bl = min(intersections, key=lambda p: p[0] - p[1])

    # 定义目标矩形大小
    width_a = np.linalg.norm(np.array(tr) - np.array(tl))
    width_b = np.linalg.norm(np.array(br) - np.array(bl))
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(np.array(tr) - np.array(br))
    height_b = np.linalg.norm(np.array(tl) - np.array(bl))
    max_height = max(int(height_a), int(height_b))

    # 透视变换后的目标点
    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    # 计算透视变换矩阵并应用变换
    rect = np.array([tl, tr, br, bl], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    transformed = cv2.warpPerspective(image, M, (max_width, max_height))

    return transformed


# 读取图像并进行处理
def pic_solve(image, id,min_radius, max_radius):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(
        gray, 125, 255, cv2.THRESH_BINARY_INV
    )  # 125 为阈值，可根据实际情况调整

    # 使用霍夫圆变换检测圆形
    circles = cv2.HoughCircles(
        binary,
        cv2.HOUGH_GRADIENT, # 使用梯度法检测圆
        dp=1.2, # 累加器分辨率与图像分辨率的反比
        minDist=40, # 圆心之间的最小距离
        param1=50, #这个参数是canny检测的高阈值，参数越大，边缘检测越弱，检测到的边缘越少
        param2=30, #这个参数是canny检测的低阈值，高于这个阈值的边缘才会被检测到
        minRadius=min_radius, # 最小半径
        maxRadius=max_radius, # 最大半径
    )

    # 初始化填涂结果存储
    results = []
    threshold = 50

    if circles is not None:
        circles = circles[0]

        for idx, circle in enumerate(circles):
            x, y, r = int(circle[0]), int(circle[1]), int(circle[2])
            if y > 500:
                continue

            circle_mask = np.zeros_like(binary)
            cv2.circle(circle_mask, (x, y), r, 255, -1)

            mean_val = cv2.mean(binary, mask=circle_mask)[0]
            filled = mean_val > threshold

            results.append(
                {
                    "X": x,
                    "Y": y,
                    "Radius": r,
                    "Status": "Filled" if filled else "Not Filled",
                }
            )

            color = (0, 255, 0) if filled else (0, 0, 255)
            cv2.circle(image, (x, y), r, color, 2)
            text = "Filled" if filled else "Empty"
            cv2.putText(
                image, text, (x - 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

    # 展示结果
    cv2.imwrite(f"output/image{id}.png", image)
    cv2.imwrite(f"output/transformed_image{id}.png", image)
    cv2.imwrite(f"output/binary_image{id}.png", binary)

    # cv2.waitKey(0)

    return results , image


def process_answers(results, num_options=5, x_tolerance=20):
    # 根据X值将结果分成左右两列
    # 获取图像宽度
    width = max(r["X"] for r in results) / 2
    left_column = [r for r in results if r["X"] < width]
    # 假设图像宽度中间值为400，可根据实际情况调整
    right_column = [r for r in results if r["X"] >= width]

    def process_column(column):
        column = sorted(column, key=lambda r: (r["Y"], r["X"]))
        questions = []
        question = []
        last_y = None

        for res in column:
            if last_y is not None and abs(res["Y"] - last_y) > x_tolerance:
                questions.append(question)
                question = []

            question.append(res)
            last_y = res["Y"]

        if question:
            questions.append(question)

        return questions

    left_questions = process_column(left_column)
    right_questions = process_column(right_column)

    # 合并两列的结果
    answers = {}
    answer_options = ["A", "B", "C", "D", "E"]
    for idx, question in enumerate(left_questions + right_questions):
        question = sorted(question, key=lambda r: r["X"])
        filled = [q["Status"] == "Filled" for q in question[:num_options]]
        answers[f"Question {idx + 1}"] = (
            answer_options[filled.index(True)] if True in filled else "No Answer"
        )

    return answers


class ImageProcessorApp:
    def __init__(self, master):
        self.master = master
        master.title("Student ID and Answer Processor")

        self.label = Label(master, text="Welcome to the Image Processor")
        self.label.pack()

        self.load_button = Button(master, text="Load Image", command=self.load_image)
        self.load_button.pack()

        self.answer_text = Text(master, height=10, width=50)
        self.answer_text.pack()

        # 添加复选框
        self.recognize_id_var = BooleanVar(value=False)
        self.id_checkbox = Checkbutton(master, text="Recognize Student ID", variable=self.recognize_id_var)
        self.id_checkbox.pack()

        # 添加最小和最大半径的输入框
        self.min_radius_label = Label(master, text="Min Radius:")
        self.min_radius_label.pack()
        self.min_radius_entry = Entry(master)
        self.min_radius_entry.pack()
        self.min_radius_entry.insert(0, "20")  # 默认值

        self.max_radius_label = Label(master, text="Max Radius:")
        self.max_radius_label.pack()
        self.max_radius_entry = Entry(master)
        self.max_radius_entry.pack()
        self.max_radius_entry.insert(0, "40")  # 默认值

        self.image_labels = []  # 用于存储图像标签

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            for label in self.image_labels:
                label.destroy()
            self.image_labels.clear()

            image = cv2.imread(file_path)
            transformed_image = perspective_transform(image)

            # 获取用户输入的半径值
            min_radius = int(self.min_radius_entry.get())
            max_radius = int(self.max_radius_entry.get())

            results, final_image = pic_solve(transformed_image, 1, min_radius, max_radius)  # 传递半径

            self.display_image(final_image, "Processed Image")

            # 根据复选框状态决定是否识别学号
            if self.recognize_id_var.get():
                student_id = student_id_get(image)
                print(f"Student ID: {student_id}")
            else:
                student_id = "Not Recognized"

            # 获取答案并显示
            results = process_answers(results)  # 确保在pic_solve中保存了结果
            self.display_answers(results, student_id)

    def display_answers(self, answers, id):
        """在文本框中显示答案"""
        self.answer_text.delete(1.0, 'end')  # 清空文本框
        answer_text = "\n".join(f"{q}: {ans}" for q, ans in answers.items())
        answer_text = "\n".join([f"Student ID: {id}", answer_text])
        self.answer_text.insert('end', answer_text)  # 插入答案

    def display_image(self, image, title):
        if image is None:
            messagebox.showerror("Error", "Image could not be loaded.")
            return

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width, _ = image.shape
        image_pil = Image.fromarray(image)
        image_pil = image_pil.resize((width // 2, height // 2), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(image_pil)

        label = Label(self.master)
        label.config(image=img_tk)
        label.image = img_tk
        label.pack()

        self.image_labels.append(label)  # 将新标签添加到列表中

if __name__ == "__main__":
    root = Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
    
    # test_count = 7
    # for id in range(7, test_count + 1):
    #     image_name = f"images/test{id}.png"
    #     image = cv2.imread(image_name)
    #     student_id_get(image)
    #     transformed_image = perspective_transform(image)
    #     results = pic_solve(transformed_image, id)

    #     print(f"Answers for image {id}:")
    #     answers = process_answers(results)
    #     for q, ans in answers.items():
    #         print(f"{q}: {ans}")
