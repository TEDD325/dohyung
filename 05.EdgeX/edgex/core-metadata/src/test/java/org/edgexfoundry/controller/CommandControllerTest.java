/*******************************************************************************
 * Copyright 2016-2017 Dell Inc.
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
 * @microservice: support-logging
 * @author: Jim White, Dell
 * @version: 1.0.0
 *******************************************************************************/

package org.edgexfoundry.controller;

import static org.edgexfoundry.test.data.CommandData.newTestInstance;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.mockito.Matchers.any;
import static org.mockito.Mockito.when;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;

import org.edgexfoundry.controller.impl.CommandControllerImpl;
import org.edgexfoundry.dao.CommandRepository;
import org.edgexfoundry.dao.DeviceProfileDao;
import org.edgexfoundry.domain.meta.Command;
import org.edgexfoundry.domain.meta.DeviceProfile;
import org.edgexfoundry.exception.controller.DataValidationException;
import org.edgexfoundry.exception.controller.LimitExceededException;
import org.edgexfoundry.exception.controller.NotFoundException;
import org.edgexfoundry.exception.controller.ServiceException;
import org.edgexfoundry.test.category.RequiresNone;
import org.edgexfoundry.test.data.CommandData;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.springframework.data.domain.Sort;

@Category(RequiresNone.class)
public class CommandControllerTest {

  private static final String LIMIT_PROPERTY = "maxLimit";
  private static final int MAX_LIMIT = 100;

  private static final String TEST_ID = "123";
  private static final String TEST_ERR_MSG = "test message";

  @InjectMocks
  private CommandControllerImpl controller;

  @Mock
  private CommandRepository repos;

  @Mock
  private DeviceProfileDao profileDao;

  private Command cmd;

  @Before
  public void setup() throws Exception {
    MockitoAnnotations.initMocks(this);
    setControllerMAXLIMIT(MAX_LIMIT);
    cmd = newTestInstance();
    cmd.setId(TEST_ID);
  }

  @Test
  public void testCommand() {
    when(repos.findOne(TEST_ID)).thenReturn(cmd);
    assertEquals("Command returned is not as expected", cmd, controller.command(TEST_ID));
  }

  @Test(expected = NotFoundException.class)
  public void testCommandNotFound() {
    controller.command(TEST_ID);
  }

  @Test(expected = ServiceException.class)
  public void testCommandException() {
    when(repos.findOne(TEST_ID)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.command(TEST_ID);
  }

  @Test
  public void testCommands() {
    List<Command> cmds = new ArrayList<>();
    cmds.add(cmd);
    when(repos.findAll(any(Sort.class))).thenReturn(cmds);
    when(repos.count()).thenReturn(1L);
    List<Command> commands = controller.commands();
    assertEquals("Number of commands returned does not matched expected number", 1,
        commands.size());
    assertEquals("Command returned is not as expected", cmd, commands.get(0));
  }

  @Test(expected = LimitExceededException.class)
  public void testCommandsMaxLimit() {
    List<Command> cmds = new ArrayList<>();
    cmds.add(cmd);
    when(repos.count()).thenReturn(1000L);
    controller.commands();
  }

  @Test(expected = ServiceException.class)
  public void testCommandsException() {
    List<Command> cmds = new ArrayList<>();
    cmds.add(cmd);
    when(repos.findAll(any(Sort.class))).thenThrow(new RuntimeException(TEST_ERR_MSG));
    when(repos.count()).thenReturn(1L);
    controller.commands();
  }

  @Test
  public void testCommandsForName() {
    List<Command> cmds = new ArrayList<>();
    cmds.add(cmd);
    when(repos.findByName(CommandData.TEST_CMD_NAME)).thenReturn(cmds);
    List<Command> commands = controller.commandForName(CommandData.TEST_CMD_NAME);
    assertEquals("Number of commands returned does not matched expected number", 1,
        commands.size());
    assertEquals("Command returned is not as expected", cmd, commands.get(0));
  }

  @Test(expected = ServiceException.class)
  public void testCommandForNameException() {
    when(repos.findByName(CommandData.TEST_CMD_NAME)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.commandForName(CommandData.TEST_CMD_NAME);
  }

  @Test
  public void testAdd() {
    when(repos.save(cmd)).thenReturn(cmd);
    assertEquals("Command ID returned is not the value expected", TEST_ID, controller.add(cmd));
  }

  @Test(expected = ServiceException.class)
  public void testAddWithNull() {
    controller.add(null);
  }

  @Test(expected = ServiceException.class)
  public void testAddServiceException() {
    when(repos.save(cmd)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.add(cmd);
  }

  @Test
  public void testUpdate() {
    when(repos.findOne(TEST_ID)).thenReturn(cmd);
    assertTrue("Command was not updated", controller.update(cmd));
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateNotFoundException() {
    when(repos.findOne(TEST_ID)).thenReturn(null);
    controller.update(cmd);
  }

  @Test(expected = ServiceException.class)
  public void testUpdateWithNull() {
    controller.update(null);
  }

  @Test
  public void testDelete() {
    when(repos.findOne(TEST_ID)).thenReturn(cmd);
    assertTrue("Command was not deleted", controller.delete(TEST_ID));
  }

  @Test(expected = NotFoundException.class)
  public void testDeleteNotFound() {
    when(repos.findOne(TEST_ID)).thenReturn(null);
    controller.delete(TEST_ID);
  }

  @Test(expected = ServiceException.class)
  public void testDeleteDaoFails() {
    when(repos.findOne(TEST_ID)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.delete(TEST_ID);
  }

  @Test(expected = DataValidationException.class)
  public void testDeleteWithAssociatedProfiles() {
    List<DeviceProfile> profiles = new ArrayList<>();
    profiles.add(new DeviceProfile());
    when(profileDao.getAssociatedProfilesForCommand(cmd)).thenReturn(profiles);
    when(repos.findOne(TEST_ID)).thenReturn(cmd);
    controller.delete(TEST_ID);
  }

  private void setControllerMAXLIMIT(int newLimit) throws Exception {
    Class<?> controllerClass = controller.getClass();
    Field temp = controllerClass.getDeclaredField(LIMIT_PROPERTY);
    temp.setAccessible(true);
    temp.set(controller, newLimit);
  }

}
