import logging
import requests

def get_llm_insights(findings,llm_config):
    # Geting more insights and recommendation on the detection using the defined LLM
    enriched_findings = []
    current=1
    count=len(findings)
    for finding in findings:
        logging.info('> Getting AI advice {}/{}'.format(current,count))
        current+=1
        finding['ai_advice'] = ''
        llm_url=llm_config['url']
        data = {
                "model": llm_config['model'],
                "prompt": (
                    "{} {}".format(llm_config['prompt'],finding['log_line'])
                ),
                "stream": False,
        }
        response = requests.post(llm_url, json=data)
        finding['ai_advice'] = response.json()['response']
        enriched_findings.append(finding)
    return enriched_findings