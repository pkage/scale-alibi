import json
import boto3
from botocore.config import Config

print('Loading function')

s3 = boto3.client('s3', config=Config(s3={'use_dualstack_endpoint': True}))


def lambda_handler(event, context):
    # print('Received event: ' + json.dumps(event, indent=2))

    # Get the object from the event and show its content type
    # bucket = event['Records'][0]['s3']['bucket']['name']
    bucket = 'pkage'
    key = 'scale-alibi/data' + event['rawPath']
    
    try:
        presigned_url = s3.generate_presigned_url('get_object', Params={'Bucket': bucket, 'Key': key})
        
        response = {}
        response['statusCode'] = 302
        response['headers'] = {'Location': presigned_url}
        data = {}
        response['body'] = json.dumps(data)
        return response
        
    except Exception as e:
        print(e)
        print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(key, bucket))
        
        response = {}
        response['statusCode'] = 404
        data = {}
        response['body']= 'no such object'
        return response

