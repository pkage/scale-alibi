# lambdas

These are some AWS Lambdas to help manage the dataset.

## `scale-alibi-data-signer.py`

Presigns S3 objects so that the scale-alibi dataset can be downloaded, and makes sure to use the [dual-stack endpoints](https://aws.amazon.com/blogs/networking-and-content-delivery/dual-stack-ipv6-architectures-for-aws-and-hybrid-networks/) for IPv6 support.

Needs the `AmazonS3ReadOnlyAccess` IAM role, and should be assigned to a function URL.

IAM roles:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:Get*",
                "s3:List*",
                "s3:Describe*",
                "s3-object-lambda:Get*",
                "s3-object-lambda:List*"
            ],
            "Resource": "*"
        }
    ]
}
```

Usage:

```
curl -L https://<function-url>.on.aws/sar_tiles.pmtile -o sar_tiles.pmtile
```


