import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

smtp = smtplib.SMTP_SSL('smtp.naver.com', 465)
smtp.ehlo()  # say Hello
# smtp.starttls()  # TLS 사용시 필요
smtp.login('@.com', '@')

# msg = MIMEText('본문 테스트 메시지')
msg = MIMEMultipart()
part = MIMEText('본문 테스트 메시지')
msg.attach(part)
path = '../evaluate_result/for_email_transfer/BCH_10_25_1_0.1_param.pickle'
msg['Subject'] = '테스트'
msg['To'] = 'ulujo_dohk@naver.com'
with open(path, 'rb') as f:
    part = MIMEBase("application", "octet-stream")
    part.set_payload(f.read())    #payload: osi 7-layers
    encoders.encode_base64(part)  #base64 encoding: 영상, 이미지 파일을 문자열 형태로 변환
    part.add_header('Content-Disposition', 'attachment', filename=path)
    msg.attach(part)

smtp.sendmail('dhgdohk@naver.com', 'ulujo_dohk@naver.com', msg.as_string())
smtp.quit()