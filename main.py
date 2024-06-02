from flask import Flask, render_template, request,flash,session
import mysql.connector
from werkzeug.utils import secure_filename
import os
import io
import base64
from Detection import detection_img

app = Flask(__name__)
app.secret_key = "abc"

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/admin")
def admin():
    return render_template("admin.html")

@app.route("/admin_login_check",methods =["GET", "POST"])
def admin_login_check():
    uid = request.form.get("unm")
    pwd = request.form.get("pwd")
    if uid=="admin" and pwd=="admin":
        return render_template("admin_home.html")
    else:
        return render_template("admin.html",msg="Invalid Credentials")

@app.route('/ahome')
def ahome():
    return render_template('admin_home.html')

'''@app.route('/uhome')
def uhome():
    return render_template('user_home.html')'''

@app.route("/evaluations")
def evaluations():
    return render_template("evaluations.html")

@app.route("/user")
def user():
    return render_template("user.html")

@app.route("/userlogin_check",methods =["GET", "POST"])
def userlogin_check():
        uid = request.form.get("unm")
        pwd = request.form.get("pwd")
        con, cur = database()
        sql = "select count(*) from user_reg where user_name='" + uid + "' and password='" + pwd + "'"
        cur.execute(sql)
        res = cur.fetchone()[0]
        if res > 0:
            session['uid'] = uid
            con, cur = database()
            uid = session['uid']
            qry = "select * from user_reg where user_name= '" + uid + " ' "
            cur.execute(qry)
            vals = cur.fetchall()
            for values in vals:
                name = values[0]
                print(name)

            return render_template("user_home.html",name=name)
        else:

            return render_template("user.html",msg="Invalid Credentials")
        return ""

@app.route("/user_reg")
def user_reg():
    return render_template("user_reg.html")

@app.route("/user_reg_store",methods =["GET", "POST"])
def user_reg_store():
    name = request.form.get('name')
    uid = request.form.get('uid')
    pwd = request.form.get('pwd')
    email = request.form.get('email')
    mno = request.form.get('mno')
    con, cur = database()
    sql = "select count(*) from user_reg where user_name='" + uid + "'"
    cur.execute(sql)
    res = cur.fetchone()[0]
    if res > 0:
        return render_template("user_reg.html", messages="User Id already exists..!")
    else:
        sql = "insert into user_reg values(%s,%s,%s,%s,%s)"
        values = (name, uid, pwd, email, mno)
        cur.execute(sql,values)
        con.commit()
        return render_template("user.html", messages="Registered Successfully..! Login Here.")
    return ""

@app.route("/user_home")
def user_home():
    con, cur = database()
    uid = session['uid']
    qry = "select * from user_reg where user_name= '" + uid + " ' "
    cur.execute(qry)
    vals = cur.fetchall()
    for values in vals:
        name = values[0]
        print(name)
    return render_template('user_home.html',name=name)

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route("/detection2",methods =["GET", "POST"])
def detection2():
    image = request.files['file']
    imgdata = secure_filename(image.filename)
    filename=image.filename
    filelist = [ f for f in os.listdir("testimg") ]
    #print(filelist)
    for f in filelist:
        os.remove(os.path.join("testimg", f))

    image.save(os.path.join("testimg", imgdata))
    image_path="..\\covid\\testimg\\"+filename
    result=detection_img(image_path)
    print("result.....",result)
    return render_template("result.html", res=result)
#DATABASE CONNECTION
def database():
    con = mysql.connector.connect(host="127.0.0.1", user='root', password="root", database="covid")
    cur = con.cursor()
    return con, cur

if __name__ == '__main__':
    app.run(host="localhost", port=5657, debug=True)