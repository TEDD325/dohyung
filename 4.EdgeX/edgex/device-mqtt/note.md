    {
        "id": "5b97f11544d00aaaf3e8d40f",
        "commands": [
            {
                "id": "5b97f11544d00aaaf3e8d40c",
                "created": 1536684309059,
                "modified": 1536684309059,
                "origin": 0,
                "name": "motor_angle",
                "get": {
                    "path": "/api/v1/device/{deviceId}/motor_angle",
                    "responses": [
                        {
                            "code": "200",
                            "description": "Angle of Motor",
                            "expectedValues": [
                                "motor_angle"
                            ]
                        },
                        {
                            "code": "503",
                            "description": "service unavailable",
                            "expectedValues": []
                        }
                    ]
                },
                "put": null
            },
            {
                "id": "5b97f11544d00aaaf3e8d40d",
                "created": 1536684309060,
                "modified": 1536684309060,
                "origin": 0,
                "name": "motor_power",
                "get": null,
                "put": {
                    "path": "/api/v1/device/{deviceId}/motor_power",
                    "responses": [
                        {
                            "code": "200",
                            "description": "Set the power of motor.",
                            "expectedValues": [
                                "motor_power"
                            ]
                        },
                        {
                            "code": "503",
                            "description": "service unavailable",
                            "expectedValues": []
                        }
                    ],
                    "parameterNames": [
                        "motor_power"
                    ]
                }
            },
            {
                "id": "5b97f11544d00aaaf3e8d40e",
                "created": 1536684309062,
                "modified": 1536684309062,
                "origin": 0,
                "name": "motor_status",
                "get": {
                    "path": "/api/v1/device/{deviceId}/motor_status",
                    "responses": [
                        {
                            "code": "200",
                            "description": "Set the snapshot duration.",
                            "expectedValues": [
                                "motor_status"
                            ]
                        },
                        {
                            "code": "503",
                            "description": "service unavailable",
                            "expectedValues": []
                        }
                    ]
                },
                "put": null
            }
        ]
    }