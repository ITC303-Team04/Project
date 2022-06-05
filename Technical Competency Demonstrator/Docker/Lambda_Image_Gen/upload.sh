# Build the docker image
sudo docker build -t road-seg .

# Create a ECR repository
aws ecr create-repository --repository-name road-seg-test --image-scanning-configuration scanOnPush=true --region ap-southeast-2 

# Tag the image to match the repository name
sudo docker tag road-seg:latest 548951197192.dkr.ecr.ap-southeast-2.amazonaws.com/road-seg-test:latest

# Register docker to ECR
aws ecr get-login-password --region ap-southeast-2 | sudo docker login --username AWS --password-stdin 548951197192.dkr.ecr.ap-southeast-2.amazonaws.com

# Push the image to ECR
sudo docker push 548951197192.dkr.ecr.ap-southeast-2.amazonaws.com/road-seg-test:latest
