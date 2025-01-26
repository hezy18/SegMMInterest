from email.header import Header
from email.mime.text import MIMEText
from email.utils import formataddr
import smtplib


def send_email(to_addr, subject, text=""):
    username = "scsegpu@163.com"
    password = "XTTCUQCCXXEYIVBG"
    from_addr = "scsegpu@163.com"
    # to_addr = "knjingwang@gmail.com"

    # subject = f"[SG] {hostname} GPU#{gpu_id}:{gpu_info['name']}"
    # text = (
    #     pd.DataFrame.from_dict(gpu_info, orient="index").to_string(header=False)
    #     + f"\nCaptured Memory: {get_pid_info()} Mb"
    # )

    message = MIMEText(text, "plain", "utf-8")
    message["From"] = formataddr((str(Header("GPU Cluster", "utf-8")), username))
    message["To"] = to_addr
    message["Subject"] = Header(subject, "utf-8")

    # print(message.as_string())

    smtp_server = smtplib.SMTP_SSL("smtp.163.com", 465)
    smtp_server.login(username, password)
    smtp_server.sendmail(from_addr, to_addr, message.as_string())
    smtp_server.close()
