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
 * @microservice: core-metadata
 * @author: Jim White, Dell
 * @version: 1.0.0
 *******************************************************************************/

package org.edgexfoundry.dao.integration;

import static org.edgexfoundry.test.data.ProfileData.TEST_LABELS;
import static org.edgexfoundry.test.data.ProfileData.TEST_MAUFACTURER;
import static org.edgexfoundry.test.data.ProfileData.TEST_MODEL;
import static org.edgexfoundry.test.data.ProfileData.TEST_PROFILE_NAME;
import static org.edgexfoundry.test.data.ProfileData.checkTestData;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.List;

import org.edgexfoundry.Application;
import org.edgexfoundry.dao.CommandRepository;
import org.edgexfoundry.dao.DeviceProfileRepository;
import org.edgexfoundry.domain.meta.Command;
import org.edgexfoundry.domain.meta.DeviceProfile;
import org.edgexfoundry.test.category.RequiresMongoDB;
import org.edgexfoundry.test.category.RequiresSpring;
import org.edgexfoundry.test.category.RequiresWeb;
import org.edgexfoundry.test.data.CommandData;
import org.edgexfoundry.test.data.ProfileData;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.SpringApplicationConfiguration;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;
import org.springframework.test.context.web.WebAppConfiguration;

@RunWith(SpringJUnit4ClassRunner.class)
@SpringApplicationConfiguration(classes = Application.class)
@WebAppConfiguration("src/test/resources")
@Category({RequiresMongoDB.class, RequiresSpring.class, RequiresWeb.class})
public class DeviceProfileRepositoryTest {

  @Autowired
  private DeviceProfileRepository repos;

  @Autowired
  private CommandRepository cmdRepos;

  private String id;

  /**
   * Create and save an instance of the DeviceProfile before each test Note: the before method tests
   * the save operation of the Repository
   */
  @Before
  public void createTestData() {
    DeviceProfile profile = ProfileData.newTestInstance();
    Command cmd = CommandData.newTestInstance();
    cmdRepos.save(cmd);
    List<Command> commands = new ArrayList<Command>();
    commands.add(cmd);
    profile.setCommands(commands);
    repos.save(profile);
    id = profile.getId();
    assertNotNull("new test Device Profile has no identifier", id);
  }

  @After
  public void cleanup() {
    repos.deleteAll();
    cmdRepos.deleteAll();
  }

  @Test
  public void testFindOne() {
    DeviceProfile profile = repos.findOne(id);
    assertNotNull("Find one returns no device profile", profile);
    checkTestData(profile, id);
  }

  @Test
  public void testFindOneWithBadId() {
    DeviceProfile profile = repos.findOne("foo");
    assertNull("Find one returns device profile with bad id", profile);
  }

  @Test
  public void testFindAll() {
    List<DeviceProfile> profiles = repos.findAll();
    assertEquals("Find all not returning a list with one device profile", 1, profiles.size());
    checkTestData(profiles.get(0), id);
  }

  @Test
  public void testFindByName() {
    DeviceProfile profile = repos.findByName(TEST_PROFILE_NAME);
    assertNotNull("Find by name returns no Device Profile", profile);
    checkTestData(profile, id);
  }

  @Test
  public void testFindByNameWithBadName() {
    DeviceProfile profile = repos.findByName("badname");
    assertNull("Find by name returns device profile with bad name", profile);
  }

  @Test
  public void testFindByLabel() {
    List<DeviceProfile> profiles = repos.findByLabelsIn(TEST_LABELS[0]);
    assertEquals("Find by labels returned no DeviceProfile", 1, profiles.size());
    checkTestData(profiles.get(0), id);
  }

  @Test
  public void testFindByLabelWithBadLabel() {
    List<DeviceProfile> profiles = repos.findByLabelsIn("foolabel");
    assertTrue("Find by labels returns device profile with bad label", profiles.isEmpty());
  }

  @Test
  public void testFindByManufacturer() {
    List<DeviceProfile> profiles = repos.findByManufacturer(TEST_MAUFACTURER);
    assertEquals("Find by manufacturer returns no Device Profile", 1, profiles.size());
    checkTestData(profiles.get(0), id);
  }

  @Test
  public void testFindByManufacturerWithBadManufacturer() {
    List<DeviceProfile> profiles = repos.findByManufacturer("badmanufacturer");
    assertTrue("Find by manufacturer returns device profile with bad manufacturer",
        profiles.isEmpty());
  }

  @Test
  public void testFindByModel() {
    List<DeviceProfile> profiles = repos.findByModel(TEST_MODEL);
    assertEquals("Find by model returns no Device Profile", 1, profiles.size());
    checkTestData(profiles.get(0), id);
  }

  @Test
  public void testFindByModelWithBadModel() {
    List<DeviceProfile> profiles = repos.findByManufacturer("badmodel");
    assertTrue("Find by model returns device profile with bad model", profiles.isEmpty());
  }

  @Test
  public void testFindByManufacturerOrModel() {
    List<DeviceProfile> profiles = repos.findByManufacturerOrModel("badmanufacturer", TEST_MODEL);
    assertEquals("Find by manufacturer or model returns no Device Profile", 1, profiles.size());
    checkTestData(profiles.get(0), id);
    profiles = repos.findByManufacturerOrModel(TEST_MAUFACTURER, "badmodel");
    assertEquals("Find by manufacturer or model returns no Device Profile", 1, profiles.size());
    checkTestData(profiles.get(0), id);
  }

  @Test
  public void testFindByManufacterOrModelWithBadManufactureAndModel() {
    List<DeviceProfile> profiles = repos.findByManufacturerOrModel("badmanufacturer", "badmodel");
    assertTrue("Find by manufactur or model returns device profile with bad manufacturer and model",
        profiles.isEmpty());
  }

  @Test(expected = DuplicateKeyException.class)
  public void testDeviceProfileWithSameName() {
    DeviceProfile profile = new DeviceProfile();
    profile.setName(TEST_PROFILE_NAME);
    repos.save(profile);
    fail("Should not have been able to save the device profile with a duplicate name");
  }

  @Test
  public void testUpdate() {
    DeviceProfile profile = repos.findOne(id);
    // check that create and modified timestamps are the same
    assertEquals("Modified and created timestamps should be equal after creation", profile.getModified(),
        profile.getCreated());
    profile.setDescription("new description");
    repos.save(profile);
    // reread device profile
    DeviceProfile profile2 = repos.findOne(id);
    assertEquals("Device profile was not updated appropriately", "new description",
        profile2.getDescription());
    assertNotEquals(
        "after modification, modified timestamp still the same as the device profile's create timestamp",
        profile2.getModified(), profile2.getCreated());
  }

  @Test
  public void testDelete() {
    DeviceProfile profile = repos.findOne(id);
    repos.delete(profile);
    assertNull("Device profile not deleted", repos.findOne(id));
  }

}
