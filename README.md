# Binary-Morphology
1. Chuỗi tham số dòng lệnh là:
main.py -i <input_file> -o <output_file> -p <mor_operator> -t <wait_key_time>
Với các tham số:
- Input_file
Đường dẫn tuyệt đối của ảnh đầu vào
- Output_file
Đường dẫn tuyệt đối của ảnh đầu ra
- Mor_operator
Mã toán tử hình thái học được áp dụng
- Wait_time_key
Thời gian chờ show các cửa sổ ảnh

2. Mor_operator ứng với các toán tử khi nhập tham số dòng lệnh:
- Erode operator - erode
- Dilate operator- dilate
- Open operator - open
- Close operator - close
- Hit or miss operator - hitmiss
- Thin operator - thin
- Boundary extract operator - boundary_extract
- Hole filling - hole_fill
- Extract connected components - acc
- Convex Hull - convex_hull
- Thickening - thicken
- Skeletons - skeleton

