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

import static org.edgexfoundry.test.data.ServiceData.TEST_SERVICE_NAME;
import static org.edgexfoundry.test.data.ServiceData.checkTestData;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.List;

import org.edgexfoundry.Application;
import org.edgexfoundry.dao.AddressableRepository;
import org.edgexfoundry.dao.DeviceServiceRepository;
import org.edgexfoundry.domain.meta.Addressable;
import org.edgexfoundry.domain.meta.DeviceService;
import org.edgexfoundry.test.category.RequiresMongoDB;
import org.edgexfoundry.test.category.RequiresSpring;
import org.edgexfoundry.test.category.RequiresWeb;
import org.edgexfoundry.test.data.AddressableData;
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
public class DeviceServiceRepositoryTest {

  @Autowired
  private DeviceServiceRepository repos;

  @Autowired
  private AddressableRepository addrRepos;

  private String id;
  private String addressableId;

  /**
   * Create and save an instance of the DeviceService before each test Note: the before method tests
   * the save operation of the Repository
   */
  @Before
  public void creatTestData() {
    Addressable addr = AddressableData.newTestInstance();
    addrRepos.save(addr);
    addressableId = addr.getId();
    DeviceService service = ServiceData.newTestInstance();
    service.setAddressable(addr);
    repos.save(service);
    id = service.getId();
    assertNotNull("new test Device Service has no identifier", id);
  }

  /**
   * Clean up of the unit test Note: clean up also tests the delete operation of the repository
   */
  @After
  public void destroyTestData() {
    repos.deleteAll();
    addrRepos.deleteAll();
  }

  @Test
  public void testFindOne() {
    DeviceService service = repos.findOne(id);
    assertNotNull("Find one returns no device service", service);
    checkTestData(service, id);
  }

  @Test
  public void testFindOneWithBadId() {
    DeviceService service = repos.findOne("foo");
    assertNull("Find one returns device service with bad id", service);
  }

  @Test
  public void testFindAll() {
    List<DeviceService> services = repos.findAll();
    assertEquals("Find all not returning a list with one device service", 1, services.size());
    checkTestData(services.get(0), id);
  }

  @Test
  public void testFindByName() {
    DeviceService service = repos.findByName(TEST_SERVICE_NAME);
    assertNotNull("Find by name returns no Device Service ", service);
    checkTestData(service, id);
  }

  @Test
  public void testFindByNameWithBadName() {
    DeviceService service = repos.findByName("badname");
    assertNull("Find by name returns device service with bad name", service);
  }

  @Test
  public void testFindByLabel() {
    List<DeviceService> services = repos.findByLabelsIn(ServiceData.TEST_LABELS[0]);
    assertEquals("Find by labels returned no DeviceService", 1, services.size());
    checkTestData(services.get(0), id);
  }

  @Test
  public void testFindByLabelWithBadLabel() {
    List<DeviceService> services = repos.findByLabelsIn("foolabel");
    assertTrue("Find by labels returns device service with bad label", services.isEmpty());
  }

  @Test
  public void testFindByAddressable() {
    List<DeviceService> services = repos.findByAddressable(addrRepos.findOne(addressableId));
    assertEquals("Find by addressable returned no Device Service", 1, services.size());
    checkTestData(services.get(0), id);
  }

  @Test
  public void testFindByAddressableWithBadAddressable() {
    Addressable addr = AddressableData.newTestInstance();
    addr.setId("abc");
    List<DeviceService> services = repos.findByAddressable(addr);
    assertTrue("Find by addressable returns services with bad addressable", services.isEmpty());
  }

  @Test(expected = DuplicateKeyException.class)
  public void testDeviceWithSameName() {
    DeviceService service = new DeviceService();
    service.setName(TEST_SERVICE_NAME);
    repos.save(service);
    fail("Should not have been able to save the service with a duplicate name");
  }


  @Test
  public void testUpdate() {
    DeviceService service = repos.findOne(id);
    // check that create and modified timestamps are the same
    assertEquals("Modified and created timestamps should be equal after creation",
        service.getModified(), service.getCreated());
    service.setDescription("new description");
    repos.save(service);
    // reread device
    DeviceService service2 = repos.findOne(id);
    assertEquals("Device service was not updated appropriately", "new description",
        service2.getDescription());
    assertNotEquals(
        "after modification, modified timestamp still the same as the device service's create timestamp",
        service2.getModified(), service2.getCreated());
  }

  @Test
  public void testDelete() {
    DeviceService service = repos.findOne(id);
    repos.delete(service);
    assertNull("Device service not deleted", repos.findOne(id));
  }

}
