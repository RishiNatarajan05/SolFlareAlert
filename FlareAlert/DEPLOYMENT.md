# FlareAlert Deployment Guide

This guide covers deploying the FlareAlert solar flare prediction system to AWS and local development.

## Prerequisites

- AWS CLI configured with appropriate permissions
- Docker and Docker Compose
- Python 3.9+
- Node.js 16+

## Local Development

### Using Docker Compose

1. **Start all services:**
```bash
docker-compose up -d
```

2. **Access the application:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

3. **View logs:**
```bash
docker-compose logs -f
```

4. **Stop services:**
```bash
docker-compose down
```

### Manual Setup

1. **Backend Setup:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

2. **Frontend Setup:**
```bash
cd frontend
npm install
npm start
```

3. **Database Setup:**
```bash
# Using PostgreSQL locally
createdb flarealert
# Or use the provided SQLite database
```

## AWS Deployment

### 1. Prepare Environment Variables

Create a `.env` file in the project root:
```env
DATABASE_PASSWORD=your_secure_password_here
AWS_REGION=us-east-1
ENVIRONMENT=dev
```

### 2. Deploy Infrastructure

```bash
# Make deployment script executable
chmod +x aws/deploy.sh

# Deploy to dev environment
export DATABASE_PASSWORD=your_secure_password_here
./aws/deploy.sh dev us-east-1
```

### 3. Deploy Frontend to S3 + CloudFront

```bash
# Build frontend for production
cd frontend
npm run build

# Create S3 bucket for frontend
aws s3 mb s3://flarealert-frontend-$(aws sts get-caller-identity --query Account --output text)

# Upload frontend
aws s3 sync build/ s3://flarealert-frontend-$(aws sts get-caller-identity --query Account --output text) --delete

# Configure bucket for static website hosting
aws s3 website s3://flarealert-frontend-$(aws sts get-caller-identity --query Account --output text) \
    --index-document index.html \
    --error-document index.html
```

### 4. Update Frontend Configuration

Update the frontend environment variables to point to your deployed API:

```env
REACT_APP_API_URL=https://your-api-gateway-url.execute-api.us-east-1.amazonaws.com/dev
REACT_APP_WS_URL=wss://your-api-gateway-url.execute-api.us-east-1.amazonaws.com/dev
```

## Infrastructure Components

### AWS Resources Created

1. **VPC with Public/Private Subnets**
   - Public subnets for load balancers
   - Private subnets for Lambda and RDS

2. **RDS PostgreSQL Database**
   - Multi-AZ deployment (production)
   - Automated backups
   - Security groups

3. **Lambda Function**
   - FastAPI wrapped with Mangum
   - VPC access for database connectivity
   - Environment variables for configuration

4. **API Gateway**
   - REST API with proxy integration
   - CORS enabled
   - Custom domain support (optional)

5. **S3 Bucket**
   - Model storage with versioning
   - Private access only
   - Lifecycle policies

6. **IAM Roles and Policies**
   - Lambda execution role
   - S3 access permissions
   - VPC access permissions

### Security Features

- Database in private subnets
- Lambda in private subnets
- Security groups with minimal access
- S3 bucket with public access blocked
- IAM roles with least privilege

## Monitoring and Logging

### CloudWatch Logs

Lambda function logs are automatically sent to CloudWatch:
```bash
aws logs describe-log-groups --log-group-name-prefix "/aws/lambda/dev-flarealert-api"
```

### Database Monitoring

RDS provides built-in monitoring:
- CPU utilization
- Memory usage
- Connection count
- Storage metrics

### API Gateway Monitoring

Monitor API usage and errors:
- Request count
- 4xx/5xx error rates
- Latency metrics

## Scaling and Performance

### Lambda Configuration

- Memory: 512MB (adjustable)
- Timeout: 30 seconds
- Concurrent executions: 1000 (default)

### Database Scaling

- RDS instance type: db.t3.micro (dev) / db.t3.small (prod)
- Storage: 20GB (auto-scaling enabled)
- Read replicas (optional for production)

### Frontend Optimization

- React build optimization
- CDN distribution via CloudFront
- Static asset caching

## Backup and Recovery

### Database Backups

- Automated daily backups
- 7-day retention (configurable)
- Point-in-time recovery

### Application Backups

- S3 versioning for models
- Git repository for code
- Environment configuration in CloudFormation

## Troubleshooting

### Common Issues

1. **Lambda Cold Start**
   - Use provisioned concurrency
   - Optimize package size
   - Use connection pooling

2. **Database Connection Issues**
   - Check security groups
   - Verify VPC configuration
   - Test connectivity from Lambda

3. **CORS Errors**
   - Configure API Gateway CORS
   - Update frontend URLs
   - Check preflight requests

### Debug Commands

```bash
# Test Lambda function
aws lambda invoke --function-name dev-flarealert-api --payload '{}' response.json

# Check API Gateway logs
aws logs describe-log-groups --log-group-name-prefix "API-Gateway-Execution-Logs"

# Test database connectivity
aws rds describe-db-instances --db-instance-identifier dev-flarealert-db
```

## Cost Optimization

### Development Environment

- Use t3.micro instances
- Disable Multi-AZ
- Minimal backup retention

### Production Environment

- Use t3.small or larger
- Enable Multi-AZ for high availability
- Longer backup retention
- CloudFront for global distribution

## Security Best Practices

1. **Secrets Management**
   - Use AWS Secrets Manager for database passwords
   - Rotate credentials regularly
   - Use IAM roles instead of access keys

2. **Network Security**
   - Keep resources in private subnets
   - Use security groups with minimal access
   - Enable VPC Flow Logs

3. **Application Security**
   - Input validation
   - Rate limiting
   - HTTPS everywhere
   - Regular security updates

## Support and Maintenance

### Regular Tasks

- Monitor CloudWatch metrics
- Review and rotate credentials
- Update dependencies
- Review security groups
- Test backup and recovery

### Emergency Procedures

1. **Database Issues**
   - Check RDS status
   - Review CloudWatch logs
   - Consider failover to read replica

2. **Lambda Issues**
   - Check function logs
   - Verify environment variables
   - Test function locally

3. **API Gateway Issues**
   - Check deployment status
   - Verify integration settings
   - Test endpoints directly
