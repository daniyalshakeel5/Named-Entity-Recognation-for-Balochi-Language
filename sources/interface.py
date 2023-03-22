from tkinter import *
import predict
import pickle

with open ('Trained_Model.pkl', 'rb') as f:
    model = pickle.load(f)



def extract():
    error = """"Nothing Found", Please Enter Something"""
    input_text = text1.get()
    prediction, _ = model.predict([input_text])


    text2.delete("1.0", END)

    if input_text == "":
        text2.insert(END, error)
    else:
        text1.delete(0, END)
        text2.insert(END, str(prediction))


window = Tk()
window.title("Balochi Named Entity Extractor")
window.geometry("500x500")

text1 = Entry(window,
              font=("Times New Roman", 12),
              width=80,
              bd=0.5,
              relief=SOLID,
              highlightcolor="Black",
              )
text1.clipboard_clear()

button = Button(window,
                bd=0,
                text="Extract",
                width=19,
                pady=10,
                background="Gray",
                fg="white",
                command=extract
                )

frame = Frame(window)

text2 = Text(frame,
             font=("Time New Roma", 12),
             width=80,
             height=15,
             fg="red",
             bd=0.5,
             relief=SOLID,
             highlightcolor="Black",
             )

text2.insert(END, "Extracted Text Comes Here")

text1.pack(pady=20, padx=20)
button.pack()
frame.pack(padx=20, pady=20)
v = Scrollbar(frame, orient='vertical')
v.pack(side=RIGHT, fill='y')
v.config(command=text2.yview)
text2.pack()

window.mainloop()
