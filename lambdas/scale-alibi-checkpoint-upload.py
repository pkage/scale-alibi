import boto3
from botocore.config import Config

print('Loading function')

s3 = boto3.client('s3', config=Config(s3={'use_dualstack_endpoint': True}))


def lambda_handler(event, context):
    # print('Received event: ' + json.dumps(event, indent=2))

    # Get the object from the event and show its content type
    # bucket = event['Records'][0]['s3']['bucket']['name']
    bucket = 'pkage'
    key = 'scale-alibi' + event['rawPath']
    
    try:
        presigned_url = s3.generate_presigned_url('put_object', Params={'Bucket': bucket, 'Key': key})
        
        return presigned_url
        # response = {}
        # response['statusCode'] = 302
        # response['headers'] = {'Location': presigned_url}
        # response['body'] = None
        # return response
        
    except Exception as e:
        print(e)
        print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(key, bucket))
        
        response = {}
        response['statusCode'] = 404
        response['body']= 'no such object'
        return response

