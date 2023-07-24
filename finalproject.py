import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



def main():
    df = pd.read_csv('py4ai_score.csv')
    df.fillna(0, inplace = True)

    tab1, tab2, tab3, tab4 = st.tabs(["Student List", "Data Visualization", "Regression", "Classification"])
    # Filter Data
    def class_classify(PYTHON_CLASS):
        if '114' in PYTHON_CLASS:
            return 'Group 114'
        if '115' in PYTHON_CLASS:
            return 'Group 115'
    df['CLASS_GROUP'] = df['PYTHON_CLASS'].apply(class_classify)

    def shift_classify(PYTHON_CLASS):
        if '-S' in PYTHON_CLASS:
            return 'Morning Class'
        if '-C' in PYTHON_CLASS:
            return 'Afternoon Class'
    df['CLASS_SHIFT'] = df['PYTHON_CLASS'].apply(shift_classify)

    def gifted_classify(CLASS):
        if 'C' in CLASS:
            return 'LỚP CHUYÊN'
        else:
            return 'LỚP THƯỜNG'
    df['GIFTED_GROUP'] = df['CLASS'].apply(gifted_classify)

    def pass_classify(GPA):
        if GPA>=6.5:
            return 'Pass'
        else:
            return 'Fail'
    df['PASS_MC'] = df['GPA'].apply(pass_classify)

    def grade_classify(CLASS):
        if '10' in CLASS:
            return 'Grade 10'
        if '11' in CLASS:
            return 'Grade 11'
        if '12' in CLASS:
            return 'Grade 12' 
    df['GRADE_LEVEL'] = df['CLASS'].apply(grade_classify)

    def detailed_gift_classify(CLASS):
        if 'CTIN' in CLASS:
            return 'CHUYÊN TIN'
        if 'CL' in CLASS:
            return 'CHUYÊN LÝ'
        if 'CH' in CLASS:
            return 'CHUYÊN HÓA'
        if 'CA' in CLASS:
            return 'CHUYÊN ANH'
        if 'CV' in CLASS:
            return 'CHUYÊN VĂN'
        if 'CT' in CLASS:
            return 'CHUYÊN TOÁN'
        else:
            return 'LỚP THƯỜNG'
    df['GIFTED_CLASS'] = df['CLASS'].apply(detailed_gift_classify)
    px.pie(df, names = 'GIFTED_CLASS')

    with tab1:
    # Data Table and Filter
        st.title("Danh sách học sinh")
        col1, col2, col3, col4 = st.columns(4)
    # For Gender Selection
        gender_options = ["All"] + list(df["GENDER"].unique())
        selected_gender = col1.radio("Theo giới tính:", gender_options)
    # For Class Group
        class_options = ["All"] + list(df["CLASS_GROUP"].unique())
        selected_class = col2.radio("Theo khối lớp:", class_options)
    #For Gifted Class
        gifted_options = ["All"] + list(df["GIFTED_GROUP"].unique())
        gifted_selection = col3.radio("Theo chuyên ban:", gifted_options)
    # For Shift Class
        shift_options = ["All"] + list(df['CLASS_SHIFT'].unique())
        selected_shift = col4.radio("Theo ca học:", shift_options)

    #Filter data following conditions:
        filtered_data = df.copy()

        if selected_gender != "All":
            filtered_data = filtered_data[filtered_data["GENDER"] == selected_gender]

        if selected_class != "All":
            filtered_data = filtered_data[filtered_data["CLASS_GROUP"] == selected_class]

        if gifted_selection != "All":
            filtered_data = filtered_data[filtered_data["GIFTED_GROUP"] == gifted_selection]

        if selected_shift != "All":
            filtered_data = filtered_data[filtered_data["CLASS_SHIFT"] == selected_shift]

        st.dataframe(filtered_data)

    with tab2:

        # Gender ratio
        gender_ratio = px.pie(df, names = 'GENDER')
        st.subheader("Tỷ lệ học sinh: Nam vs Nữ")
        st.plotly_chart(gender_ratio)

        python_class_ratio = px.pie(df, names = 'CLASS_GROUP')
        st.subheader("Tỷ lệ học sinh các lớp Python")
        st.plotly_chart(python_class_ratio)

        shift_class_ratio = px.pie(df, names = 'CLASS_SHIFT')
        st.subheader("Tỷ lệ học sinh theo ca học")
        st.plotly_chart(shift_class_ratio)

        class_level_ratio = px.pie(df, names = 'GRADE_LEVEL')
        st.subheader("Tỷ lệ học sinh các khối")
        st.plotly_chart(class_level_ratio)

        gifted_class_ratio = px.pie(df, names = 'GIFTED_CLASS')
        st.subheader("Tỷ lệ học sinh các lớp chuyên ban")
        st.plotly_chart(gifted_class_ratio)

        # Grade by session
        session_columns = np.delete(df.columns[4:16], 10)
        selected_session = st.selectbox("Chọn session", session_columns)    
        
        st.subheader(f"Phân bố điểm của Session {selected_session} theo giới tính")
        fig = px.box(df, x='GENDER', y = selected_session)
        st.plotly_chart(fig)

        # Grade by Class-Group
        st.subheader(f"Phân bố điểm của Session {selected_session} theo khối lớp")
        fig2 = px.box(df, x = 'CLASS_GROUP', y = selected_session)
        st.plotly_chart(fig2)

        # Grade by Class-Shift
        st.subheader(f"Phân bố điểm của Session {selected_session} theo ca học")
        fig3 = px.box(df, x = 'CLASS_SHIFT', y = selected_session)
        st.plotly_chart(fig3)

        
        st.subheader(f"Phân bố điểm của Session {selected_session} theo ban")
        fig4 = px.box(df, x = 'GIFTED_GROUP', y = selected_session)
        st.plotly_chart(fig4)

        # Pass ratio for MC Class  
        st.subheader("Tỉ lệ đậu lớp MC")
        mc_pass_ratio = px.pie(df, names = 'PASS_MC')
        st.plotly_chart(mc_pass_ratio)

        # Reg ratio for MC Class
        st.subheader("Tỉ lệ đăng ký học tiếp lớp MC")
        mc_reg_ratio = px.pie(df, names = 'REG-MC4AI')
        st.plotly_chart(mc_reg_ratio)

    with tab3:

        # KMEAN Classification
        X = np.array(df[['GPA']])
        
        kmeans = KMeans(n_clusters=3, n_init='auto')
        kmeans.fit(X)
        df['Cluster'] = kmeans.labels_

        st.title('Phân nhóm học sinh theo điểm GPA')

        # Tạo màu sắc tương ứng với từng nhóm
        colors = ['#FF0000', '#00FF00', '#0000FF']

        # Tạo một đối tượng Figure
        fig = go.Figure()

        # Thêm scatter 3D vào đối tượng Figure
        for cluster in range(3):
            clustered_data = df[df['Cluster'] == cluster]
            fig.add_trace(go.Scatter3d(
                x=clustered_data['S6'],
                y=clustered_data['S10'],
                z=clustered_data['GPA'],
                mode='markers',
                marker=dict(color=colors[cluster]),
                name=f'Cluster {cluster}'
            ))

        # Hiển thị biểu đồ trong Streamlit
        st.plotly_chart(fig)

        # Hiển thị danh sách học sinh theo từng nhóm đã phân loại        
        avg_gpa = df.groupby('Cluster')['GPA'].mean()
        st.write("Danh sách học sinh theo phân nhóm:")
        for cluster in range(3):            
            st.write(f"Nhóm {cluster+1} - Điểm trung bình GPA: {avg_gpa[cluster]:.2f}")
            st.write(df[df['Cluster'] == cluster])

        # Dự đoán và phân loại
        df['AVG_HW'] = df[['S1', 'S2', 'S3', 'S4', 'S5', 'S7', 'S8', 'S9']].mean(axis=1)
        # Chọn các điểm số cần dự đoán và các điểm số đầu vào
        final_X = df[['AVG_HW', 'S6']]
        final_y = df['S10']

        gpa_X = df[['S6', 'S10']]
        gpa_y = df['GPA']

        pass_fail_X = df[['S6', 'AVG_HW']]
        pass_fail_y = df['PASS_MC']

        option = st.selectbox('Chọn loại dự đoán:', ('Dự đoán điểm cuối kỳ', 'Dự đoán GPA', 'Dự đoán Pass/Fail'))

        if option == 'Dự đoán điểm cuối kỳ':
            # Hiển thị form để nhập điểm trung bình của homework và midterm
            homework_avg = st.number_input('Điểm trung bình homework:', min_value=df['AVG_HW'].min(), max_value=df['AVG_HW'].max(), step=0.1)
            midterm = st.number_input('Điểm midterm:', min_value=df['S6'].min(), max_value=df['S6'].max(), step=0.1)

            # Tạo mô hình hồi quy tuyến tính
            model_final = LinearRegression()
            model_final.fit(final_X, final_y)

            # Dự đoán điểm cuối kỳ dựa trên điểm trung bình của homework và midterm
            new_data = pd.DataFrame({'AVG_HW': [homework_avg], 'S6': [midterm]})
            predicted_final = model_final.predict(new_data)

            # Hiển thị kết quả dự đoán
            st.write(f"Dự đoán điểm cuối kỳ: {predicted_final[0]}")
        
        elif option == 'Dự đoán GPA':
            # Hiển thị form để nhập điểm midterm và final
            midterm = st.number_input('Điểm midterm:', min_value=df['S6'].min(), max_value=df['S6'].max(), step=0.1)
            final = st.number_input('Điểm final:', min_value=df['S10'].min(), max_value=df['S10'].max(), step=0.1)

            # Tạo mô hình hồi quy tuyến tính
            model_gpa = LinearRegression()
            model_gpa.fit(gpa_X, gpa_y)      

            # Dự đoán GPA dựa trên điểm midterm và final
            new_score = pd.DataFrame({'S6': [midterm], 'S10': [final]})
            predicted_gpa = model_gpa.predict(new_score)

            # Hiển thị kết quả dự đoán
            st.write(f"Dự đoán GPA: {predicted_gpa[0]}")

        elif option == 'Dự đoán Pass/Fail':
            # Hiển thị form để nhập điểm midterm và điểm trung bình của homework
            midterm = st.number_input('Điểm midterm:', min_value=df['S6'].min(), max_value=df['S6'].max(), step=0.1)
            homework_avg = st.number_input('Điểm trung bình homework:', min_value=df['AVG_HW'].min(), max_value=df['AVG_HW'].max(), step=0.1)

            # Tạo mô hình Logistic Regression
            model_pf = LogisticRegression()
            model_pf.fit(pass_fail_X, pass_fail_y)

            # Dự đoán Pass/Fail dựa trên điểm midterm và điểm trung bình của homework
            new_data = pd.DataFrame({'S6': [midterm], 'AVG_HW': [homework_avg]})
            predicted_pass_fail = model_pf.predict(new_data)

            # Hiển thị kết quả phân loại
            st.write(f"Dự báo Pass/Fail: {predicted_pass_fail[0]}")
    
    with tab4:

        st.subheader("Phân loại theo 2 đặc trưng")

        X = df[['S6', 'AVG_HW']]
        y = df['PASS_MC']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression()
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        st.write("Độ chính xác của mô hình:", accuracy)
        weights = model.coef_[0]
        bias = model.intercept_[0]
        w1, w2 = weights

        plt.figure(figsize=(5, 5))
        plt.scatter(X['AVG_HW'], X['S6'], c=y.map({'Pass': 1, 'Fail': 0}))
        plt.xlabel('AVG_HW')
        plt.ylabel('S6')
        x1 = np.array([0, 10])
        x2 = -(bias + w1 * x1) / w2
        plt.plot(x1, x2, c='r')
        st.pyplot(plt)

        st.subheader("Phân loại theo 3 đặc trưng")

        X = df[['AVG_HW', 'S6', 'S10']].values.copy()
        y = df['PASS_MC'].values.copy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression()
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        st.write("Độ chính xác của mô hình:", accuracy)

        w1, w2, w3 = model.coef_[0]
        b = model.intercept_[0]

        x1 = np.array([X[:,0].min(), X[:,0].max()]) 
        y1 = np.array([X[:,1].min(), X[:,1].max()])

        xx, yy = np.meshgrid(x1, y1)
        xy = np.c_[xx.ravel(), yy.ravel()]
        z = (-b - w1*xy[:,0] - w2*xy[:,1])/w3

        z = z.reshape(xx.shape)

        fig = go.Figure(data=[go.Scatter3d(x=X[:,0], y=X[:,1], z=X[:,2], mode='markers', marker_color=df['PASS_MC'].map({'Pass': 1, 'Fail': 0})),
                            go.Surface(x=x1, y=y1, z=z)])
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()