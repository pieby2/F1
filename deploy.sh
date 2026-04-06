#!/bin/bash
# ============================================================
# F1 Race Predictor — Full AWS Deployment Script
# ============================================================
# Usage:
#   ./deploy.sh setup    — Create ECR repos + EKS cluster (one-time)
#   ./deploy.sh build    — Build & push Docker images to ECR
#   ./deploy.sh deploy   — Apply K8s manifests to EKS
#   ./deploy.sh all      — setup + build + deploy
#   ./deploy.sh destroy  — Tear down everything
# ============================================================

set -euo pipefail

# ─── CONFIGURATION (edit these) ───
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
EKS_CLUSTER_NAME="f1-predictor"
ECR_REPO_API="f1-predictor-api"
ECR_REPO_WEB="f1-predictor-web"
K8S_NAMESPACE="f1-predictor"
NODE_TYPE="${NODE_TYPE:-t3.large}"
NODE_COUNT="${NODE_COUNT:-2}"
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_TAG="${IMAGE_TAG:-latest}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🏎️  F1 Predictor Deployment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Region:    ${AWS_REGION}"
echo "Account:   ${AWS_ACCOUNT_ID}"
echo "Cluster:   ${EKS_CLUSTER_NAME}"
echo "Registry:  ${ECR_REGISTRY}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ─── STEP 1: Setup AWS Infrastructure ───
setup() {
    echo ""
    echo "🔧 Step 1: Creating ECR repositories..."
    aws ecr create-repository --repository-name "${ECR_REPO_API}" --region "${AWS_REGION}" 2>/dev/null || echo "  → ${ECR_REPO_API} already exists"
    aws ecr create-repository --repository-name "${ECR_REPO_WEB}" --region "${AWS_REGION}" 2>/dev/null || echo "  → ${ECR_REPO_WEB} already exists"

    echo ""
    echo "🔧 Step 2: Creating EKS cluster (takes 15-20 min)..."
    eksctl create cluster \
        --name "${EKS_CLUSTER_NAME}" \
        --region "${AWS_REGION}" \
        --version 1.29 \
        --nodegroup-name workers \
        --node-type "${NODE_TYPE}" \
        --nodes "${NODE_COUNT}" \
        --nodes-min 1 \
        --nodes-max 3 \
        --managed

    echo ""
    echo "🔧 Step 3: Installing AWS Load Balancer Controller..."
    # Create IAM service account
    eksctl create iamserviceaccount \
        --cluster="${EKS_CLUSTER_NAME}" \
        --namespace=kube-system \
        --name=aws-load-balancer-controller \
        --role-name="AmazonEKSLoadBalancerControllerRole-${EKS_CLUSTER_NAME}" \
        --attach-policy-arn=arn:aws:iam::${AWS_ACCOUNT_ID}:policy/AWSLoadBalancerControllerIAMPolicy \
        --approve

    helm repo add eks https://aws.github.io/eks-charts
    helm repo update
    helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
        -n kube-system \
        --set clusterName="${EKS_CLUSTER_NAME}" \
        --set serviceAccount.create=false \
        --set serviceAccount.name=aws-load-balancer-controller

    echo "✅ Infrastructure setup complete!"
}

# ─── STEP 2: Build & Push Docker Images ───
build() {
    echo ""
    echo "🐳 Logging in to ECR..."
    aws ecr get-login-password --region "${AWS_REGION}" | \
        docker login --username AWS --password-stdin "${ECR_REGISTRY}"

    echo ""
    echo "🐳 Building backend image..."
    docker build \
        --target backend \
        -t "${ECR_REGISTRY}/${ECR_REPO_API}:${IMAGE_TAG}" \
        .

    echo ""
    echo "🐳 Building frontend image..."
    docker build \
        -t "${ECR_REGISTRY}/${ECR_REPO_WEB}:${IMAGE_TAG}" \
        ./web

    echo ""
    echo "🐳 Pushing images to ECR..."
    docker push "${ECR_REGISTRY}/${ECR_REPO_API}:${IMAGE_TAG}"
    docker push "${ECR_REGISTRY}/${ECR_REPO_WEB}:${IMAGE_TAG}"

    echo "✅ Images pushed!"
}

# ─── STEP 3: Deploy to Kubernetes ───
deploy() {
    echo ""
    echo "☸️  Updating kubeconfig..."
    aws eks update-kubeconfig --name "${EKS_CLUSTER_NAME}" --region "${AWS_REGION}"

    echo ""
    echo "☸️  Applying Kubernetes manifests..."

    # Create namespace
    kubectl apply -f k8s/namespace.yaml

    # Storage
    kubectl apply -f k8s/pvc.yaml

    # Update image references in deployment manifests
    TMPDIR=$(mktemp -d)
    cp k8s/backend-deployment.yaml "${TMPDIR}/backend.yaml"
    cp k8s/frontend-deployment.yaml "${TMPDIR}/frontend.yaml"

    sed -i "s|<ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/f1-predictor-api:latest|${ECR_REGISTRY}/${ECR_REPO_API}:${IMAGE_TAG}|g" "${TMPDIR}/backend.yaml"
    sed -i "s|<ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/f1-predictor-web:latest|${ECR_REGISTRY}/${ECR_REPO_WEB}:${IMAGE_TAG}|g" "${TMPDIR}/frontend.yaml"

    kubectl apply -f "${TMPDIR}/backend.yaml"
    kubectl apply -f "${TMPDIR}/frontend.yaml"
    kubectl apply -f k8s/ingress.yaml
    kubectl apply -f k8s/hpa.yaml

    rm -rf "${TMPDIR}"

    echo ""
    echo "☸️  Waiting for rollout..."
    kubectl rollout status deployment/f1-api -n "${K8S_NAMESPACE}" --timeout=300s
    kubectl rollout status deployment/f1-web -n "${K8S_NAMESPACE}" --timeout=120s

    echo ""
    echo "☸️  Getting ALB URL..."
    sleep 15
    ALB_URL=$(kubectl get ingress f1-predictor-ingress -n "${K8S_NAMESPACE}" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "pending")

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🏁 DEPLOYMENT COMPLETE!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Dashboard URL: http://${ALB_URL}"
    echo "API URL:       http://${ALB_URL}/api/health"
    echo ""
    echo "Useful commands:"
    echo "  kubectl get pods -n ${K8S_NAMESPACE}"
    echo "  kubectl logs -f deployment/f1-api -n ${K8S_NAMESPACE}"
    echo "  kubectl get ingress -n ${K8S_NAMESPACE}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# ─── DESTROY ───
destroy() {
    echo ""
    echo "⚠️  This will delete the entire EKS cluster and all resources!"
    read -p "Are you sure? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "Cancelled."
        exit 0
    fi

    echo "Deleting K8s resources..."
    kubectl delete -f k8s/ingress.yaml --ignore-not-found
    kubectl delete -f k8s/hpa.yaml --ignore-not-found
    kubectl delete -f k8s/frontend-deployment.yaml --ignore-not-found
    kubectl delete -f k8s/backend-deployment.yaml --ignore-not-found
    kubectl delete -f k8s/pvc.yaml --ignore-not-found
    kubectl delete -f k8s/namespace.yaml --ignore-not-found

    echo "Deleting EKS cluster..."
    eksctl delete cluster --name "${EKS_CLUSTER_NAME}" --region "${AWS_REGION}"

    echo "Deleting ECR repositories..."
    aws ecr delete-repository --repository-name "${ECR_REPO_API}" --region "${AWS_REGION}" --force 2>/dev/null || true
    aws ecr delete-repository --repository-name "${ECR_REPO_WEB}" --region "${AWS_REGION}" --force 2>/dev/null || true

    echo "✅ All resources destroyed!"
}

# ─── MAIN ───
case "${1:-}" in
    setup)   setup ;;
    build)   build ;;
    deploy)  deploy ;;
    all)     setup && build && deploy ;;
    destroy) destroy ;;
    *)
        echo "Usage: $0 {setup|build|deploy|all|destroy}"
        echo ""
        echo "  setup    — Create ECR repos + EKS cluster (one-time, ~20 min)"
        echo "  build    — Build & push Docker images to ECR"
        echo "  deploy   — Apply K8s manifests to EKS"
        echo "  all      — Run setup + build + deploy"
        echo "  destroy  — Tear down all AWS resources"
        exit 1
        ;;
esac
