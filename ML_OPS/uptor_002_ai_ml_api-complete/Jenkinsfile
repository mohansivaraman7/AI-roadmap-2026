pipeline {
    agent any

    options {
        timestamps()
        disableConcurrentBuilds()
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Docker Sanity Check') {
            steps {
                sh '''
                    docker --version
                    docker compose version
                '''
            }
        }

        stage('Clean Previous State') {
            steps {
                // In case a previous build on this agent left containers/
                // volumes/images behind, start from a known-clean slate
                // rather than risk building on top of something stale
                // (exactly the kind of "why is this still failing" trap
                // we hit doing this by hand).
                sh 'docker compose down -v --remove-orphans || true'
            }
        }

        stage('Build Images') {
            steps {
                // Builds both images used by this project: Dockerfile.mlflow
                // (the mlflow service) and Dockerfile (shared by train + api).
                sh 'docker compose build'
            }
        }

        stage('Start MLflow') {
            steps {
                sh '''
                    docker compose up -d mlflow

                    echo "Waiting for MLflow to become ready..."
                    ready=0
                    for i in $(seq 1 30); do
                        if curl -s -o /dev/null http://127.0.0.1:5001/; then
                            ready=1
                            echo "MLflow is up"
                            break
                        fi
                        sleep 1
                    done

                    if [ "$ready" -ne 1 ]; then
                        echo "MLflow did not start in time"
                        docker compose logs mlflow || true
                        exit 1
                    fi
                '''
            }
        }

        stage('Train Model') {
            steps {
                // One-off job: trains the model, logs the run to MLflow,
                // and writes house_model.pkl back onto the agent via the
                // bind mount defined in docker-compose.yml.
                sh 'docker compose run --rm train'
            }
        }

        stage('Start API & Smoke Test') {
            steps {
                sh '''
                    docker compose up -d api

                    echo "Waiting for API to become ready..."
                    ready=0
                    for i in $(seq 1 30); do
                        if curl -s -o /dev/null http://127.0.0.1:5000/; then
                            ready=1
                            echo "API is up"
                            break
                        fi
                        sleep 1
                    done

                    if [ "$ready" -ne 1 ]; then
                        echo "API did not start in time"
                        docker compose logs api || true
                        exit 1
                    fi

                    response=$(curl -s -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"size": 1700}')
                    echo "Response: $response"

                    echo "$response" | grep -q "predicted_price" || {
                        echo "Response did not contain predicted_price"
                        exit 1
                    }
                '''
            }
        }
    }

    post {
        always {
            // Capture logs from all three services before tearing down —
            // this is what you'd otherwise lose the moment the containers
            // stop, and it's exactly what we needed while debugging this
            // pipeline by hand.
            sh 'docker compose logs > compose.log 2>&1 || true'
            archiveArtifacts artifacts: 'house_model.pkl, compose.log', allowEmptyArchive: true

            // Tear everything down, including volumes, so the next build
            // starts clean rather than accumulating state across runs.
            sh 'docker compose down -v --remove-orphans || true'
        }
        success {
            echo 'Build, train, and smoke test succeeded.'
        }
        failure {
            echo 'Pipeline failed — check compose.log and the console output above for details.'
        }
    }
}
