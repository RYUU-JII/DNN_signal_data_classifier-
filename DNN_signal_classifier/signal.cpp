#include QThread
#include QSerialPort
#include QSerialPortInfo
#include QObject
#include QDebug
#include QByteArray
#include QMap
#include QString

class NeuroPyWorker  public QThread {
    Q_OBJECT

public
    NeuroPyWorker(const QString &portName, int baudRate = 57600, int timeout = 1000, QObject parent = nullptr)
         QThread(parent), portName(portName), baudRate(baudRate), timeout(timeout), threadRun(false) {
        serial = new QSerialPort(this);
    }

    ~NeuroPyWorker() {
        stop();
        closeSerialPort();
    }

signals
    void emitSignal(QMapQString, int data);   UI로 데이터 전송
    void errorSignal(const QString &error);     에러 메시지 전송

protected
    void run() override {
        if (!openSerialPort()) {
            emit errorSignal(시리얼 연결 실패);
            return;
        }

        threadRun = true;

        while (threadRun) {
            try {
                QMapQString, int data = readPacket();
                if (!data.isEmpty()) {
                    emit emitSignal(data);
                }
            } catch (const stdexception &e) {
                emit errorSignal(QString(데이터 읽기 실패 %1).arg(e.what()));
            }
        }

        closeSerialPort();
    }

private
    QSerialPort serial;
    QString portName;
    int baudRate;
    int timeout;
    bool threadRun;

    bool openSerialPort() {
        serial-setPortName(portName);
        serial-setBaudRate(baudRate);
        serial-setDataBits(QSerialPortData8);
        serial-setParity(QSerialPortNoParity);
        serial-setStopBits(QSerialPortOneStop);
        serial-setFlowControl(QSerialPortNoFlowControl);

        if (!serial-open(QIODeviceReadOnly)) {
            qDebug()  Failed to open port  portName;
            return false;
        }

        qDebug()  portName  에 연결됨.;
        return true;
    }

    QMapQString, int readPacket() {
        QMapQString, int data;
        QByteArray payload;
        int checksum = 0;

         헤더 검사 (aa aa)
        if (!waitForHeader()) {
            throw stdruntime_error(헤더를 찾지 못했습니다.);
        }

         페이로드 읽기
        int payloadLength = readByte().toInt(nullptr, 16);
        for (int i = 0; i  payloadLength; ++i) {
            QString byteStr = readByte();
            payload.append(byteStr.toUInt(nullptr, 16));
            checksum += byteStr.toInt(nullptr, 16);
        }

         체크섬 검증
        checksum = ~checksum & 0xFF;
        if (checksum != readByte().toInt(nullptr, 16)) {
            throw stdruntime_error(체크섬 오류);
        }

        return parsePayload(payload);
    }

    bool waitForHeader() {
        while (true) {
            if (readByte() == aa && readByte() == aa) {
                return true;
            }
        }
        return false;
    }

    QString readByte() {
        if (!serial-waitForReadyRead(timeout)) {
            throw stdruntime_error(데이터 읽기 시간 초과);
        }
        QByteArray data = serial-read(1);
        return QStringnumber(data[0], 16).rightJustified(2, '0');
    }

    QMapQString, int parsePayload(const QByteArray &payload) {
        QMapQString, int data;
        int i = 0;

        try {
            while (i  payload.size()) {
                QString code = QStringnumber(payload[i], 16).rightJustified(2, '0');
                if (code == 02) {
                    ++i;
                    data[poorSignal] = payload[i];
                } else if (code == 04) {
                    ++i;
                    data[attention] = payload[i];
                } else if (code == 05) {
                    ++i;
                    data[meditation] = payload[i];
                } else if (code == 16) {
                    ++i;
                    data[blinkStrength] = payload[i];
                } else if (code == 80) {
                    i += 2;
                    int rawValue = (payload[i]  8) + payload[i + 1];
                    if (rawValue  32768) rawValue -= 65536;
                } else if (code == 83) {
                    ++i;
                    data.unite(parseEEG(payload, i));
                    i += 24;   다음 코드로 이동
                }
                ++i;
            }
        } catch (const stdexception &e) {
            throw stdruntime_error(QString(데이터 파싱 오류 %1).arg(e.what()).toStdString());
        }

        return data;
    }

    QMapQString, int parseEEG(const QByteArray &payload, int index) {
        QMapQString, int eegData;
        QStringList labels = {delta, theta, lowAlpha, highAlpha,
                              lowBeta, highBeta, lowGamma, midGamma};

        for (const QString &label  labels) {
            int val0 = payload[index];
            int val1 = payload[index + 1];
            int val2 = payload[index + 2];
            eegData[label] = (val0  16) + (val1  8) + val2;
            index += 3;
        }

        return eegData;
    }

public slots
    void stop() {
        threadRun = false;
        quit();
        wait();
        closeSerialPort();
    }

    void closeSerialPort() {
        if (serial-isOpen()) {
            serial-close();
            qDebug()  portName  포트가 닫혔습니다.;
        }
    }
};
