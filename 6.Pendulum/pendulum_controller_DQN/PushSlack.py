import json
import requests

class PushSlack:
    def __init__(self):
        # self.webhook_url = "https://hooks.slack.com/services/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"
        self.webhook_url = "https://hooks.slack.com/services/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"

    def send_message(self, username=None, message=None):
        slack_data = {'text': message}

        requests.post(
            self.webhook_url,
            data=json.dumps(slack_data),
            headers={'Content-Type': 'application/json'}
        )

if __name__ == '__main__':
    p = PushSlack()
    p.send_message(username="dohyung", message="pendulum_stopped")
