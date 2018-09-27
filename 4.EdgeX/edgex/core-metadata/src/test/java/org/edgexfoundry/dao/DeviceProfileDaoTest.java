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

package org.edgexfoundry.dao;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.mockito.Mockito.when;

import java.util.ArrayList;
import java.util.List;

import org.edgexfoundry.domain.meta.Command;
import org.edgexfoundry.domain.meta.DeviceProfile;
import org.edgexfoundry.exception.controller.DataValidationException;
import org.edgexfoundry.test.category.RequiresNone;
import org.edgexfoundry.test.data.CommandData;
import org.edgexfoundry.test.data.ProfileData;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

@Category(RequiresNone.class)
public class DeviceProfileDaoTest {

  private static final String TEST_ID = "123";

  @InjectMocks
  private DeviceProfileDao dao;

  @Mock
  private DeviceProfileRepository repos;

  @Mock
  private DeviceRepository deviceRepos;

  private DeviceProfile profile;

  @Before
  public void setup() throws Exception {
    MockitoAnnotations.initMocks(this);
    profile = ProfileData.newTestInstance();
  }

  @Test
  public void testGetByIdOrNameWithNull() {
    assertNull("Returned device profile is not null", dao.getByIdOrName(null));
  }

  @Test
  public void testGetByIdOrName() {
    when(repos.findOne(TEST_ID)).thenReturn(profile);
    profile.setId(TEST_ID);
    assertEquals("Returned device profile is not expected", profile, dao.getByIdOrName(profile));
  }

  @Test
  public void testGetByIdOrNameWithNoId() {
    when(repos.findByName(ProfileData.TEST_PROFILE_NAME)).thenReturn(profile);
    assertEquals("Returned device profile is not expected", profile, dao.getByIdOrName(profile));
  }

  @Test
  public void testCheckCommandNamesWithNewValue() {
    List<Command> commands = new ArrayList<>();
    Command command = CommandData.newTestInstance();
    Command command2 = new Command();
    Command command3 = new Command();
    command2.setName("c2");
    command3.setName("c3");
    commands.add(command);
    commands.add(command2);
    commands.add(command3);
    dao.checkCommandNames(commands, "c4");
  }

  @Test(expected = DataValidationException.class)
  public void testCheckCommandNamesWithNewValueNonUnique() {
    List<Command> commands = new ArrayList<>();
    Command command = CommandData.newTestInstance();
    Command command2 = new Command();
    Command command3 = new Command();
    command2.setName("c2");
    command3.setName("c3");
    commands.add(command);
    commands.add(command2);
    commands.add(command3);
    dao.checkCommandNames(commands, "c3");
  }

  @Test
  public void testCheckCommandNames() {
    List<Command> commands = new ArrayList<>();
    Command command = CommandData.newTestInstance();
    commands.add(command);
    dao.checkCommandNames(commands);
  }

  @Test(expected = DataValidationException.class)
  public void testCheckCommandNamesNonUnique() {
    List<Command> commands = new ArrayList<>();
    Command command = CommandData.newTestInstance();
    Command command2 = CommandData.newTestInstance();
    commands.add(command);
    commands.add(command2);
    dao.checkCommandNames(commands);
  }

  @Test
  public void testGetAssociatedProfilesForCommand() {
    List<DeviceProfile> profiles = new ArrayList<>();
    profiles.add(profile);
    Command command = CommandData.newTestInstance();
    List<Command> commands = new ArrayList<>();
    commands.add(command);
    profile.setCommands(commands);
    when(repos.findAll()).thenReturn(profiles);
    assertEquals("Expected profiles not returned", profiles,
        dao.getAssociatedProfilesForCommand(command));
  }

  @Test
  public void testGetAssociatedProfilesForCommandWithNull() {
    assertEquals("Expected profiles should be empty", 0,
        dao.getAssociatedProfilesForCommand(null).size());
  }

  @Test
  public void testCheckCommandNamesWithNull() {
    dao.checkCommandNames(null);
  }

  @Test
  public void testGetOwningServices() {
    dao.getOwningServices(profile);
  }

  @Test
  public void testGetById() {
    dao.getById(TEST_ID);
  }

  @Test
  public void testGetByName() {
    dao.getByName(ProfileData.TEST_PROFILE_NAME);
  }
}
