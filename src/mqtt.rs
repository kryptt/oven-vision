use std::fmt;
use std::time::Duration;

use rumqttc::{AsyncClient, ClientError, ConnectionError, EventLoop, MqttOptions, QoS};
use serde_json::json;
use tracing::{info, warn};

use crate::config::{DialConfig, LedConfig, MqttConfig};
use crate::detect::{DialReading, DialState};
use crate::types::LedState;

const AVAILABILITY_TOPIC: &str = "oven_vision/availability";
const DEVICE_ID: &str = "oven_vision";

#[derive(Debug)]
pub enum MqttError {
    Client(ClientError),
    Connection(ConnectionError),
}

impl fmt::Display for MqttError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Client(err) => write!(f, "MQTT client error: {err}"),
            Self::Connection(err) => write!(f, "MQTT connection error: {err}"),
        }
    }
}

impl std::error::Error for MqttError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Client(err) => Some(err),
            Self::Connection(err) => Some(err),
        }
    }
}

impl From<ClientError> for MqttError {
    fn from(err: ClientError) -> Self {
        Self::Client(err)
    }
}

impl From<ConnectionError> for MqttError {
    fn from(err: ConnectionError) -> Self {
        Self::Connection(err)
    }
}

/// Manages MQTT connection and Home Assistant sensor discovery/publishing.
pub struct MqttPublisher {
    client: AsyncClient,
    eventloop: Option<EventLoop>,
}

impl MqttPublisher {
    /// Create a new MQTT publisher from the given configuration.
    pub fn new(config: &MqttConfig) -> Result<Self, MqttError> {
        let mut opts = MqttOptions::new("oven-vision", &config.host, config.port);
        opts.set_keep_alive(Duration::from_secs(config.keepalive_secs));

        if let (Some(user), Some(pass)) = (&config.user, &config.pass) {
            opts.set_credentials(user, pass);
        }

        // Last Will and Testament: mark offline if we disconnect unexpectedly
        opts.set_last_will(rumqttc::LastWill {
            topic: AVAILABILITY_TOPIC.to_string(),
            message: "offline".into(),
            qos: QoS::AtLeastOnce,
            retain: true,
        });

        let (client, eventloop) = AsyncClient::new(opts, 10);

        Ok(Self {
            client,
            eventloop: Some(eventloop),
        })
    }

    /// Start the MQTT event loop, publish online status, and send HA discovery configs.
    pub async fn start(
        &mut self,
        dials: &[DialConfig],
        leds: &[LedConfig],
    ) -> Result<(), MqttError> {
        // Take the event loop out so we can move it into a spawned task
        let mut eventloop = self
            .eventloop
            .take()
            .expect("start() must only be called once");

        tokio::spawn(async move {
            loop {
                match eventloop.poll().await {
                    Ok(_event) => {}
                    Err(err) => {
                        warn!(%err, "MQTT event loop error, retrying");
                        tokio::time::sleep(Duration::from_secs(5)).await;
                    }
                }
            }
        });

        // Give the event loop a moment to establish the connection
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Announce online
        self.client
            .publish(AVAILABILITY_TOPIC, QoS::AtLeastOnce, true, "online")
            .await?;

        self.publish_discovery(dials, leds).await?;

        info!("MQTT publisher started, discovery configs published");
        Ok(())
    }

    /// Publish Home Assistant MQTT discovery configuration for all sensors.
    async fn publish_discovery(
        &self,
        dials: &[DialConfig],
        leds: &[LedConfig],
    ) -> Result<(), MqttError> {
        let device = json!({
            "identifiers": [DEVICE_ID],
            "name": "Kitchen Stove",
            "model": "Oven Vision v1",
            "manufacturer": "DIY"
        });

        let led_labels: Vec<&str> = leds.iter().map(|l| l.label.as_str()).collect();

        for dial in dials {
            let label = &dial.label;
            let slug = slug(label);

            // Binary sensor: on/off
            let binary_topic = format!("homeassistant/binary_sensor/oven_vision_{slug}/config");
            let binary_payload = json!({
                "name": format!("{label}"),
                "unique_id": format!("oven_vision_{slug}"),
                "state_topic": format!("oven_vision/{slug}/state"),
                "payload_on": "ON",
                "payload_off": "OFF",
                "device_class": "heat",
                "availability_topic": AVAILABILITY_TOPIC,
                "payload_available": "online",
                "payload_not_available": "offline",
                "device": device,
            });
            self.client
                .publish(&binary_topic, QoS::AtLeastOnce, true, binary_payload.to_string())
                .await?;

            // Enum sensor: heat level
            let level_topic = format!("homeassistant/sensor/oven_vision_{slug}_level/config");
            let level_payload = json!({
                "name": format!("{label} Level"),
                "unique_id": format!("oven_vision_{slug}_level"),
                "state_topic": format!("oven_vision/{slug}/level"),
                "options": ["off", "low", "medium", "high", "max"],
                "availability_topic": AVAILABILITY_TOPIC,
                "payload_available": "online",
                "payload_not_available": "offline",
                "device": device,
            });
            self.client
                .publish(&level_topic, QoS::AtLeastOnce, true, level_payload.to_string())
                .await?;

            // LED mode sensor (only if this dial has a matching LED)
            if led_labels.contains(&label.as_str()) {
                let mode_topic =
                    format!("homeassistant/sensor/oven_vision_{slug}_mode/config");
                let mode_payload = json!({
                    "name": format!("{label} Mode"),
                    "unique_id": format!("oven_vision_{slug}_mode"),
                    "state_topic": format!("oven_vision/{slug}/mode"),
                    "options": ["off", "on", "heating"],
                    "availability_topic": AVAILABILITY_TOPIC,
                    "payload_available": "online",
                    "payload_not_available": "offline",
                    "device": device,
                });
                self.client
                    .publish(&mode_topic, QoS::AtLeastOnce, true, mode_payload.to_string())
                    .await?;
            }
        }

        Ok(())
    }

    /// Publish current dial and LED states.
    pub async fn publish_states(
        &self,
        readings: &[DialReading],
        led_states: &[(String, LedState)],
    ) -> Result<(), MqttError> {
        let all_unavailable = readings
            .iter()
            .all(|r| matches!(r.state, DialState::Unavailable));

        if all_unavailable {
            self.client
                .publish(AVAILABILITY_TOPIC, QoS::AtLeastOnce, true, "offline")
                .await?;
            return Ok(());
        }

        for reading in readings {
            let slug = slug(&reading.label);

            // Binary state
            let state_str = match reading.state {
                DialState::Off | DialState::Unavailable => "OFF",
                DialState::On(_) => "ON",
            };
            self.client
                .publish(
                    format!("oven_vision/{slug}/state"),
                    QoS::AtLeastOnce,
                    false,
                    state_str,
                )
                .await?;

            // Heat level
            let level_str = match reading.state {
                DialState::Off | DialState::Unavailable => "off".to_string(),
                DialState::On(level) => level.to_string(),
            };
            self.client
                .publish(
                    format!("oven_vision/{slug}/level"),
                    QoS::AtLeastOnce,
                    false,
                    level_str,
                )
                .await?;
        }

        for (label, state) in led_states {
            let slug = slug(label);
            self.client
                .publish(
                    format!("oven_vision/{slug}/mode"),
                    QoS::AtLeastOnce,
                    false,
                    state.to_string(),
                )
                .await?;
        }

        Ok(())
    }

}

/// Convert a label to a URL/topic-safe slug (lowercase, spaces to underscores).
fn slug(label: &str) -> String {
    label.to_lowercase().replace(' ', "_")
}
