/*******************************************************************************
 * Copyright 2017 Dell Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 * @microservice: support-rulesengine
 * @author: Jim White, Dell
 * @version: 1.0.0
 *******************************************************************************/

package org.edgexfoundry.engine;

import org.apache.commons.lang3.reflect.FieldUtils;
import org.edgexfoundry.controller.CmdClient;
import org.edgexfoundry.test.category.RequiresNone;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

@Category(RequiresNone.class)
public class CommandExecutorTest {

  private static final String TEST_DEVICE = "test_device_id";
  private static final String TEST_CMD = "test_command_id";
  private static final String TEST_BODY = "test body";

  @InjectMocks
  private CommandExecutor executor;

  @Mock
  private CmdClient client;

  @Before
  public void setup() {
    MockitoAnnotations.initMocks(this);
  }

  @Test
  public void testFireCommand() {
    executor.fireCommand(TEST_DEVICE, TEST_CMD, TEST_BODY);
  }

  @Test
  public void testFireCommandException() {
    Mockito.when(client.put(TEST_DEVICE, TEST_CMD, TEST_BODY)).thenThrow(new RuntimeException());
    executor.fireCommand(TEST_DEVICE, TEST_CMD, TEST_BODY);
  }

  @Test
  public void testFireCommandNoClient() throws IllegalAccessException {
    FieldUtils.writeField(executor, "client", null, true);
    executor.fireCommand(TEST_DEVICE, TEST_CMD, TEST_BODY);
  }

}
