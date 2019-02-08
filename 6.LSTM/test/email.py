import smtplib
from email.mime.text import MIMEText

smtp = smtplib.SMTP_SSL('smtp.naver.com', 465)
smtp.ehlo()  # say Hello
# smtp.starttls()  # TLS 사용시 필요
smtp.login('@.com', '@')

msg = MIMEText('본문 테스트 메시지')
msg['Subject'] = '테스트'
msg['To'] = 'ulujo_dohk@naver.com'
smtp.sendmail('dhgdohk@naver.com', 'ulujo_dohk@naver.com', msg.as_string())

smtp.quit()