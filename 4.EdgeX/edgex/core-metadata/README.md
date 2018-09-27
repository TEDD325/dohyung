Main Author:  Jim White

Copyright 2016-17, Dell, Inc.

Metadata Micro Service - includes the device/sensor metadata database and APIs to expose the database to other services.  In particular, the device provisioning service will deposit and manage device metadata through this service.  This service may also hold and manage other configuration metadata used by other services on the gateway – such as clean up schedules, hardware configuration (Wi-Fi connection info, MQTT queues, etc.).  Non-device metadata may need to be held in a different database and/or managed by another service – depending on implementation.

