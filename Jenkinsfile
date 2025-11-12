pipeline {
    agent any

    environment {
        PROJECT_DIR = '/var/jenkins_home/workspace/EntregaFinal_cdp'
        CREDENTIALS_PATH = '/var/jenkins_home/.keys/service_account.json'
    }

    options {
        timestamps()
        buildDiscarder(logRotator(numToKeepStr: '10'))
    }

    stages {

        stage('0️⃣ - Instalar Python') {
            steps {
                echo '🐍 Verificando o instalando Python...'
                sh '''
                    apt-get update -y
                    apt-get install -y python3 python3-pip
                    python3 --version
                '''
            }
        }

        stage('2.1 - Acceder al proyecto local') {
            steps {
                echo '📦 Listando archivos del proyecto...'
                dir("${PROJECT_DIR}") {
                    sh 'ls -la'
                }
            }
        }

        stage('2.2 - Verificar estructura del proyecto') {
            steps {
                echo '🧩 Ejecutando verificación de estructura del proyecto...'
                dir("${PROJECT_DIR}/pyops") {
                    sh 'python3 check_structure.py'
                }
            }
        }

        stage('2.3 - Instalar dependencias') {
            steps {
                echo '📦 Instalando dependencias desde requirements.txt...'
                dir("${PROJECT_DIR}") {
                    sh '''
                        pip3 install --break-system-packages -r requirements.txt
                    '''
                }
            }
        }

        stage('2.4 - Ejecutar scripts principales') {
            steps {
                dir("${PROJECT_DIR}/src") {
                    echo '⚙️ Ejecutando scripts principales del pipeline...'
                    sh '''
                        export GOOGLE_APPLICATION_CREDENTIALS="${CREDENTIALS_PATH}"
                        python3 ft_engineering.py
                        python3 model_training.py
                        python3 model_evaluation.py
                    '''
                }
            }
        }

        stage('2.5 - Guardar artefactos') {
            steps {
                echo '💾 Guardando artefactos generados...'
                dir("${PROJECT_DIR}/models") {
                    archiveArtifacts artifacts: '*.pkl, *.csv, *.json', onlyIfSuccessful: true
                }
            }
        }
    }

    post {

        success {
            script {
                currentBuild.description = "✅ Ejecución exitosa del pipeline"
                echo '🎉 ============================================'
                echo '✅ Pipeline ejecutado correctamente'
                echo '📦 Dependencias instaladas y scripts completados'
                echo '💾 Artefactos guardados en /models'
                echo '==============================================='
            }
        }

        failure {
            script {
                currentBuild.description = "❌ Error en el pipeline"
                echo '💥 ============================================'
                echo '❌ Pipeline falló durante la ejecución'
                echo '🔍 Posibles causas:'
                echo '   - Python no instalado'
                echo '   - Error en BigQuery o credenciales'
                echo '   - Dependencias en requirements.txt'
                echo '   - Archivos o carpetas faltantes'
                echo '==============================================='
            }
        }

        always {
            echo "📋 Pipeline finalizado a las ${new Date()}"
        }
    }
}