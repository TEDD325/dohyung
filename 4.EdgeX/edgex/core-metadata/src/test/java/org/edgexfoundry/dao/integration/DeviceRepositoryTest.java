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

import static org.edgexfoundry.test.data.DeviceData.TEST_LABELS;
import static org.edgexfoundry.test.data.DeviceData.TEST_NAME;
import static org.edgexfoundry.test.data.DeviceData.checkTestData;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.List;

import org.edgexfoundry.Application;
import org.edgexfoundry.dao.AddressableRepository;
import org.edgexfoundry.dao.DeviceProfileRepository;
import org.edgexfoundry.dao.DeviceRepository;
import org.edgexfoundry.dao.DeviceServiceRepository;
import org.edgexfoundry.domain.meta.Addressable;
import org.edgexfoundry.domain.meta.Device;
import org.edgexfoundry.domain.meta.DeviceProfile;
import org.edgexfoundry.domain.meta.DeviceService;
import org.edgexfoundry.test.category.RequiresMongoDB;
import org.edgexfoundry.test.category.RequiresSpring;
import org.edgexfoundry.test.category.RequiresWeb;
import org.edgexfoundry.test.data.AddressableData;
import org.edgexfoundry.test.data.DeviceData;
import org.edgexfoundry.test.data.ProfileData;
import org.edgexfoundry.test.data.ServiceData;
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
public class DeviceRepositoryTest {

  @Autowired
  private DeviceRepository repos;

  @Autowired
  private DeviceServiceRepository serviceRepos;

  @Autowired
  private AddressableRepository addrRepos;

  @Autowired
  private DeviceProfileRepository profileRepos;

  private String id;
  private String serviceId;
  private String profileId;
  private String addressableId;

  /**
   * Create and save an instance of the DeviceProfile before each test Note: the before method tests
   * the save operation of the Repository
   */
  @Before
  public void createTestData() {
    DeviceService service = ServiceData.newTestInstance();
    serviceRepos.save(service);
    serviceId = service.getId();
    DeviceProfile profile = ProfileData.newTestInstance();
    profileRepos.save(profile);
    profileId = profile.getId();
    Addressable addr = AddressableData.newTestInstance();
    addrRepos.save(addr);
    addressableId = addr.getId();
    Device device = DeviceData.newTestInstance();
    device.setService(service);
    device.setProfile(profile);
    device.setAddressable(addr);
    repos.save(device);
    id = device.getId();
    assertNotNull("new test Device has no identifier", id);
  }

  @After
  public void cleanup() {
    repos.deleteAll();
    serviceRepos.deleteAll();
    profileRepos.deleteAll();
    addrRepos.deleteAll();
  }

  @Test
  public void testFindOne() {
    Device device = repos.findOne(id);
    assertNotNull("Find one returns no device", device);
    checkTestData(device, id);
  }

  @Test
  public void testFindOneWithBadId() {
    Device device = repos.findOne("foo");
    assertNull("Find one returns device with bad id", device);
  }

  @Test
  public void testFindAll() {
    List<Device> devices = repos.findAll();
    assertEquals("Find all not returning a list with one device", 1, devices.size());
    checkTestData(devices.get(0), id);
  }

  @Test
  public void testFindByName() {
    Device device = repos.findByName(TEST_NAME);
    assertNotNull("Find by name returns no Device ", device);
    checkTestData(device, id);
  }

  @Test
  public void testFindByNameWithBadName() {
    Device device = repos.findByName("badname");
    assertNull("Find by name returns device with bad name", device);
  }

  @Test
  public void testFindByLabel() {
    List<Device> devices = repos.findByLabelsIn(TEST_LABELS[0]);
    assertEquals("Find by labels returned no Device", 1, devices.size());
    checkTestData(devices.get(0), id);
  }

  @Test
  public void testFindByLabelWithBadLabel() {
    List<Device> devices = repos.findByLabelsIn("foolabel");
    assertTrue("Find by labels returns device with bad label", devices.isEmpty());
  }

  @Test
  public void testFindByService() {
    List<Device> devices = repos.findByService(serviceRepos.findOne(serviceId));
    assertEquals("Find by service returned no Device", 1, devices.size());
    checkTestData(devices.get(0), id);
  }

  @Test
  public void testFindByServiceWithBadService() {
    DeviceService service = new DeviceService();
    service.setId("abc");
    List<Device> devices = repos.findByService(service);
    assertTrue("Find by service returns device with service", devices.isEmpty());
  }

  @Test
  public void testFindByProfile() {
    List<Device> devices = repos.findByProfile(profileRepos.findOne(profileId));
    assertEquals("Find by profile returned no device", 1, devices.size());
    checkTestData(devices.get(0), id);
  }

  @Test
  public void testFindByProfileWithBadProfile() {
    DeviceProfile profile = new DeviceProfile();
    profile.setId("abc");
    List<Device> devices = repos.findByProfile(profile);
    assertTrue("Find by profile returns device with bad profile", devices.isEmpty());
  }

  @Test
  public void testFindByAddressable() {
    List<Device> devices = repos.findByAddressable(addrRepos.findOne(addressableId));
    assertEquals("Find by addressable returned no Device", 1, devices.size());
    checkTestData(devices.get(0), id);
  }

  @Test
  public void testFindByAddressableWithBadAddressable() {
    Addressable addr = AddressableData.newTestInstance();
    addr.setId("abc");
    List<Device> devices = repos.findByAddressable(addr);
    assertTrue("Find by addressable returns device with bad addressable", devices.isEmpty());
  }

  @Test(expected = DuplicateKeyException.class)
  public void testDeviceWithSameName() {
    Device device = new Device();
    device.setName(TEST_NAME);
    repos.save(device);
    fail("Should not have been able to save the device with a duplicate name");
  }

  @Test
  public void testUpdate() {
    Device device = repos.findOne(id);
    // check that create and modified timestamps are the same
    assertEquals("Modified and created timestamps should be equal after creation",
        device.getModified(), device.getCreated());
    device.setDescription("new description");
    repos.save(device);
    // reread device
    Device device2 = repos.findOne(id);
    assertEquals("Device was not updated appropriately", "new description",
        device2.getDescription());
    assertNotEquals(
        "after modification, modified timestamp still the same as the device's create timestamp",
        device2.getModified(), device2.getCreated());
  }

  @Test
  public void testDelete() {
    Device device = repos.findOne(id);
    repos.delete(device);
    assertNull("Device not deleted", repos.findOne(id));
  }

}
