/*******************************************************************************
 * Copyright 2017 Dell Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @microservice:  export-distro
 * @author: Jim White, Dell
 * @version: 1.0.0
 *******************************************************************************/
package org.edgexfoundry.messaging;

import static org.edgexfoundry.test.data.RegistrationData.TEST_ADDRESS;
import static org.edgexfoundry.test.data.RegistrationData.TEST_ADDR_NAME;
import static org.edgexfoundry.test.data.RegistrationData.TEST_PASSWORD;
import static org.edgexfoundry.test.data.RegistrationData.TEST_PORT;
import static org.edgexfoundry.test.data.RegistrationData.TEST_PROTOCOL;
import static org.edgexfoundry.test.data.RegistrationData.TEST_PUBLISHER;
import static org.edgexfoundry.test.data.RegistrationData.TEST_TOPIC;
import static org.edgexfoundry.test.data.RegistrationData.TEST_USER;
import static org.junit.Assert.*;

import org.edgexfoundry.domain.meta.Addressable;
import org.edgexfoundry.messaging.AzureMQTTSender;
import org.edgexfoundry.test.data.DeviceData;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class AzureMQTTSenderTest {

	private final String MSG_PAYLOAD = "this is a test";

	AzureMQTTSender sender;

	@Before
	public void setup() {
		Addressable addressable = new Addressable(TEST_ADDR_NAME, TEST_PROTOCOL, TEST_ADDRESS, TEST_PORT, TEST_PUBLISHER, TEST_USER, TEST_PASSWORD, TEST_TOPIC);
		sender = new AzureMQTTSender(addressable, DeviceData.TEST_NAME);
	}

	@After
	public void cleanup() {
	}

	@Test
	public void testSendMessage() {
		// connection information not setup to call Azure.  It will exercise process, but not get through.
		// This should return false
		assertFalse(sender.sendMessage(new String(MSG_PAYLOAD).getBytes()));
	}
}
