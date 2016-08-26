/**
  * Created by samay on 8/16/16.
  */
import org.apache.spark.sql.Dataset

case class RawData(
                    events: Dataset[Event],
                    appEvents: Dataset[AppEvent],
                    appLabels: Dataset[AppLabel],
                    labelCategories: Dataset[LabelCategory],
                    deviceInfo: Dataset[Device])

case class AppEvent(event_id: String, app_id: String, is_installed: Boolean, is_active: Boolean)
case class AppLabel(app_id: String, label_id: String)
case class Event(event_id: Long, device_id: Long)
case class LabelCategory(label_id: String, category: String)
case class Device(device_id: String, phone_brand: String, device_model: String)
case class Target(device_id: String, gender: String, age: String, group: String)
