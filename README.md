# RCNN-Understanding

## Giới thiệu
Năm 2012 [Ross Girshick và các cộng sự](https://arxiv.org/pdf/1311.2524.pdf) đã đề xuất phương pháp mới cho object detection - R-CNN. Sau nay nó đã trở thành tiền đề cho các phương pháp như Fast R-CNN, Faster R-CNN và Mask RCNN.

Trước RCNN để giải quyết các bài toán object detection thời bấy giờ có các phương pháp như Sliding window + Image Pyramid hay họ Deformable Part Model... R-CNN ở thời điểm ra mắt cho kết quả vượt trội so với các phương pháp trên. Các bạn có thể đọc thêm ở bài báo gốc ở link phía trên.

Cái tên R-CNN bắt nguồn từ những kỹ thuật được sử dụng trong phương pháp này đó là :
- Region proposals
- CNN

Có một số phương pháp để tạo ra các region proposals như Selective Search,... nhưng trong R-CNN tác giả sử dụng Selective Search. Trong bài báo gốc tác giả sử dụng mô hình CNN AlexNet (chiến thắng trong cuộc thi năm 2012 phân loại trên bộ dữ liệu ImageNet). Tác giả cũng so sánh performance khi dùng các backbone khác nhau. Đến thời điểm hiện tại nếu dùng các pre-trained mới hơn như ResNet chẳng hạn performance có thể được nâng lên.

Ở đây mình xin tóm tắt các bước của [selective search](https://ivi.fnwi.uva.nl/isis/publications/2013/UijlingsIJCV2013/UijlingsIJCV2013.pdf) để tạo ra các region proposals:
- Generate initial sub-segmentation, we generate many candidate regions 
- Use greedy algorithm (thuật toán tham lam) to recursively combine similar regions into larger ones (dựa trên color similarity, texture similarity, size similarity, meta similarity)
- Use the generated regions to produce the final candidate region proposals 
<img src="https://media.geeksforgeeks.org/wp-content/uploads/20200128135031/step3.PNG">

Dưới đây là mô hình AlexNet.
<img src="https://www.researchgate.net/publication/329790469/figure/fig2/AS:705721712787456@1545268576139/Simplified-illustration-of-the-AlexNet-architecture.ppm">

## Mô tả chi tiết
Bên dưới là kiến trúc của R-CNN
<img src="https://lilianweng.github.io/lil-log/assets/images/RCNN.png">

Ơ đây chúng ta đi tìm hiểu các bước hoạt động chính của R-CNN trong quá trình **training**, khi inference thì đơn giản hơn nhiều (ở đây mình tập trung vào quá trinhf training):
1. Pre-train CNN model trên ImageNet
2. Dùng selective search để lấy ra khoảng 2000 region proposals. Các vùng này có kích thước khác nhau, có thể chứa object hoặc không.
3. Các region proposals được resize lại về cùng kích thước cho phù hợp với mạng CNN (pre-trained CNN như AlexNet). Cuối cùng nhận được **warped region**.
4. Fine-tuning pre-trained CNN model dựa trên các **warped regions** cho $K+1$ classes. $K$ ở đây chính là số classes trong bộ object detection dataset. Cần cộng thêm 1 để tính đến background. 
Việc Fine tuning này giúp cải thiện performance so với việc dùng trực tiếp pre-trained model. Trong quá trình Fine tuning dùng SGD với learning rate nhỏ 0.001. Mỗi mini-batch lấy 32 positive examples và 96 negative examples (vì chú yếu là backgrounds)
- Bước 5: Train linear SVM cho mỗi class. Region proposals được đi qua CNN bên trên và trích xuất ra feature vector. Lưu tất cả feature vector cho từng class. Sau đó sẽ train **binary SVM classifier** cho từng class độc lập với nhau. 
Việc lấy positive và negative example cho từng class như sau: positive chỉ là các ground-truth boxes của class đó, negetive là các region proposals có IoU < 0.3 so với các instances của class đó. Những proposals có IoU > 0.3 so với các ground-truth của từng class bị bỏ qua.
5. Để tăng độ chính xác cho bounding box một mô hình regession đã đào tạo được sử dụng để xác định 4 offset values. Ví dụ như khi region proposal chứa người nhưng chỉ có phần thân và nửa mặt, nửa mặt còn lại không có trong region proposal đó thì offset value có thể giúp mở rộng region proposal để lấy được toàn bộ người. 


## Bounding Box Regression 
Đầu vào cho regressor là cặp $(P_i, G_i)$ - $P_i$ có 4 giá trị tương ứng $p_x, p_y, p_w, p_h$ của region proposal (tọa độ tâm, width và height), $G_i$ cũng tương tự như vậy $g_x, g_y, g_w, g_h$ nhưng cho ground-truth bounding boxes. (Cái này thực hiện sau SVM để biết thuộc class nào).

Regressor sẽ học scale-invariant transformation giữa 2 tâm và log-scale transformation giữa các width và height.

Chúng ta có thể tinh chỉnh vị trí của bounding box dựa trên các công thức sau. Nên nhớ $p_x, p_y, p_w, p_h$ là những giá trị đã biết dựa trên vị trí của region proposal. $d_x(\mathbf{p}), d_y(\mathbf{p}), d_w(\mathbf{p}), d_y(\mathbf{p})$ là những gía trị dự đoán được từ regression model.

$$\begin{aligned}
\hat{g}_x &= p_w d_x(\mathbf{p}) + p_x \\
\hat{g}_y &= p_h d_y(\mathbf{p}) + p_y \\
\hat{g}_w &= p_w \exp({d_w(\mathbf{p})}) \\
\hat{g}_h &= p_h \exp({d_h(\mathbf{p})})
\end{aligned}$$

<img src="https://lilianweng.github.io/lil-log/assets/images/RCNN-bbox-regression.png">

*Chuyển đổi giữa ground-truth box và bounding box từ region proposal*

Lợi ích của việc chuyển đổi này thay vì dùng giá trị tuyệt đối luôn vì $d_i(\mathbf{p})$ với $i \in \{ x, y, w, h \}$ có thể nhận bất kỳ giá trị nào trong khoảng (-∞, +∞). 

Dưới đây chính là label ban đầu chúng ta có:

$$\begin{aligned}
t_x &= (g_x - p_x) / p_w \\
t_y &= (g_y - p_y) / p_h \\
t_w &= \log(g_w/p_w) \\
t_h &= \log(g_h/p_h)
\end{aligned}$$

Nhìn công thức này chắc phần nào mọi người đã hiểu hơn. Bây giờ chúng ta có ground-truth box và box of the region proposal. Do đó chúng ta có thể xác định được các giá trị $t_x, t_y, t_w, t_h$. Nhiệm vụ của chúng ta đi xây dựng regression model cho 4 đại lượng này với đầu vào là feature vector của region proposal.

Xây dựng loss cho các bài toán regression này:
$$\mathcal{L}_* = \sum_{i=\{1,N\}} (t_*^{i} - d_*^{i}(\mathbf{p}))^2 + \lambda \|\mathbf{w_*}\|^2$$

Trong đó $*$ lần lượt là $x, y, w, h$.
Công thức này cần sửa lại một chút. Ở đây loss tính riêng cho từng giá trị: 2 tọa độ tâm, width và height chứ không gộp chung lại.

**Chú ý**: Việc xác định các cặp $(P_i, G_i)$ cũng rất quan trọng, không phải cặp nào cũng thỏa mãn. Nếu box of the region proposal mà quá xa ground-truth bounding box thì việc học sẽ rất khó chính xác và không có ý nghĩa nhiều lắm. Do đó cần chọn $P_i$ gần với $G_i$. Cụ thể ở đây $P_i$ được gán cho $G_i$ mà với $G_i$ đó nó có IoU cao nhất, IoU cùng phải lớn hơn 0.6 mới lấy. Các P có IoU thấp hơn 0.6 không được sử dụng.

## Một số vấn đề về object detection
### IoU - Intersection over union**

IoU được xử dụng trong bài toán object detection, để đánh giá xem bounding box dự đoán đối tượng khớp với ground truth thật của đối tượng. Nó cũng được sử dụng để lựa chọn boudingbox cho một object.
Chỉ số IoU:

- Có giá trị trong đoạn [0, 1]
- IoU càng gần 1 thì bounding box dự đoán càng gần ground-truth
![IoU](/images/IoU.png)
### Non-max suppression**

Một object có thể được phát hiện trong nhều bounding box. Để loại bỏ bớt các bounding box chỉ giữ lại duy nhất 1 cái cho mỗi object ta làm như sau:

- Lấy bounding box có probability lớn nhất 
- Lấy các bouding boxes có IoU với bounding phía trên > threshold (ví dụ 0.5)
- Loại bỏ các bounding boxes đó
- Lặp lại bước đầu tiên cho đến hết

### Vấn đề của R-CNN
- Do lấy khoảng 2000 proposal regions và phải train NN cho số regions đó nên thời gian train model lâu
- Vì lý do trên nên không thực hiện real-time được do thời gian xử lý cho mỗi test image khoảng 47 giây
- Selective search có thể đưa ra những proposal regions không được tốt.

## Tài liệu tham khảo
1. https://arxiv.org/abs/1311.2524
2. https://learnopencv.com/selective-search-for-object-detection-cpp-python/
3. https://lilianweng.github.io/lil-log/2017/12/31/object-recognition-for-dummies-part-3.html
4. https://towardsdatascience.com/understanding-object-detection-and-r-cnn-e39c16f37600
5. https://towardsdatascience.com/understanding-fast-r-cnn-and-faster-r-cnn-for-object-detection-adbb55653d97
