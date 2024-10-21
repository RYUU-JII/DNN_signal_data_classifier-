#include <QThread>
#include <QSerialPort>
#include <QObject>
#include <QDebug>
#include <QMap>
#include <QString>

class NeuroPyWorker : public QThread {
    Q_OBJECT

public:
    NeuroPyWorker(const QString& portName, int baudRate = 57600, int timeout = 1000, QObject* parent = nullptr)
        : QThread(parent), portName(portName), baudRate(baudRate), timeout(timeout) {
        serial = new QSerialPort(this);
        isRunning = false;
    }

    ~NeuroPyWorker() {
        stop()
    }

signals:
    void dataReady(QMap<QString, int> data);  // 데이터를 UI로 보내는 신호
    void errorOccurred(const QString& error); // 에러 신호

protected:
    void run() override {
        if (!openSerialPort()) {
            emit errorOccurred("포트 열기 실패.");
            return;
        }

        isRunning = true;
        while (isRunning) {
            try {
                auto packet = readPacket();
                if (!packet.isEmpty()) {
                    emit dataReady(packet);  // 데이터를 UI로 전송
                }
            }
            catch (const std::exception& e) {
                emit errorOccurred(QString("데이터 읽기 실패: %1").arg(e.what()));
            }
        }

        closeSerialPort();
    }

private:
    QSerialPort* serial;
    QString portName;
    int baudRate;
    int timeout;
    bool isRunning;

    bool openSerialPort() {
        serial->setPortName(portName);
        serial->setBaudRate(baudRate);
        serial->setDataBits(QSerialPort::Data8);
        serial->setParity(QSerialPort::NoParity);
        serial->setStopBits(QSerialPort::OneStop);
        serial->setFlowControl(QSerialPort::NoFlowControl);

        if (!serial->open(QIODevice::ReadOnly)) {
            qDebug() << "포트 열기 실패: " << portName;
            return false;
        }

        qDebug() << "포트 열림 (" << portName << ")";
        return true;
    }

    QMap<QString, int> readPacket() {
        if (!waitForHeader()) {
            throw std::runtime_error("헤더 없음");
        }

        QByteArray payload;
        int payloadLength = readByte().toInt(nullptr, 16);
        int checksum = 0;

        for (int i = 0; i < payloadLength; ++i) {
            QString byteStr = readByte();
            payload.append(byteStr.toUInt(nullptr, 16));
            checksum += byteStr.toInt(nullptr, 16);
        }

        checksum = ~checksum & 0xFF;
        if (checksum != readByte().toInt(nullptr, 16)) {
            throw std::runtime_error("체크섬 에러");
        }

        return parsePayload(payload);
    }

    bool waitForHeader() {
        while (true) {
            if (readByte() == "aa" && readByte() == "aa") {
                return true;
            }
        }
        return false;
    }

    QString readByte() {
        if (!serial->waitForReadyRead(timeout)) {
            throw std::runtime_error("데이터 수신 불가");
        }
        QByteArray data = serial->read(1);
        return QString::number(data[0], 16).rightJustified(2, '0');
    }

    QMap<QString, int> parsePayload(const QByteArray& payload) {
        QMap<QString, int> data;
        int i = 0;

        while (i < payload.size()) {
            QString code = QString::number(payload[i], 16).rightJustified(2, '0');
            if (code == "02") {
                data["poorSignal"] = payload[++i];
            }
            else if (code == "04") {
                data["attention"] = payload[++i];
            }
            else if (code == "05") {
                data["meditation"] = payload[++i];
            }
            else if (code == "16") {
                data["blinkStrength"] = payload[++i];
            }
            else if (code == "80") {
                int rawValue = (payload[++i] << 8) + payload[++i];
                if (rawValue > 32768) rawValue -= 65536;
                // row data
            }
            else if (code == "83") {
                data.unite(parseEEG(payload, ++i));
                i += 24;  // 다음으로
            }
            ++i;
        }

        return data;
    }

    QMap<QString, int> parseEEG(const QByteArray& payload, int index) {
        QMap<QString, int> eegData;
        QStringList labels = { "delta", "theta", "lowAlpha", "highAlpha",
                              "lowBeta", "highBeta", "lowGamma", "midGamma" };

        for (const QString& label : labels) {
            int value = (payload[index] << 16) + (payload[index + 1] << 8) + payload[index + 2];
            eegData[label] = value;
            index += 3;
        }

        return eegData;
    }

public slots:
    void stop() {
        isRunning = false;
        quit();
        wait()
        closeSerialPort();
    }

    void closeSerialPort() {
        if (serial->isOpen()) {
            serial->close();
            qDebug() << "포트가 닫혔습니다: " << portName;
        }
    }
};
