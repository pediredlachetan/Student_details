import pandas as pd
import pickle 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

student_data=pd.read_csv('student.csv')

gender_encode=LabelEncoder()
status_encode=LabelEncoder()

student_data['Gender']=gender_encode.fit_transform(student_data['Gender'])
student_data['Status']=status_encode.fit_transform(student_data['Status'])

features=["Percentage","Gender"]
x=student_data[features]
y=student_data['Status']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
model=LogisticRegression()
model.fit(x_train,y_train)

rollno=input("Enter student Roll no:".lower())

match=student_data[student_data["Roll.no"].str.lower()==rollno]

if match.empty:
    print("Student not found")
else:
    decode_gender=gender_encode.inverse_transform(match['Gender'])

    display_data=match.copy()
    display_data["Gender"]=decode_gender

    print("Student data:")
    print(display_data[["Roll.no","Name","Percentage","Gender"]])

student_features=match[features]
predicted=model.predict(student_features)

if predicted==0:
    print("Fail")
else:
    print("Pass")

pickle.dump(model, open('model.pkl', 'wb'))

student_data.to_csv('encoded_student_data.csv', index=False)

pickle.dump(gender_encode, open('gender_encoder.pkl', 'wb'))
pickle.dump(status_encode, open('status_encoder.pkl', 'wb'))

    
